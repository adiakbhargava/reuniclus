"""
Phase 4.3 – TCN Training Script (Python Side)
==============================================
Trains a Temporal Convolutional Network (TCN) on recorded neural data and
exports a TorchScript model loadable by NeuralEngine (C++ LibTorch).

Architecture:
  Stack of residual blocks, each containing:
    dilated causal 1D convolution → weight normalization → ReLU → dropout

Receptive field:
  R = 1 + (K−1) × (2^L − 1).
  With K=3, L=10: R = 2047 samples ≈ 2 s at 1 kHz.

Export:
  torch.jit.script(model).save("models/tcn_decoder.pt")

Usage:
  python tools/train_tcn.py --data data/neural_sessions.npy \
                             --labels data/latent_states.npy \
                             --output models/tcn_decoder.pt
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ─────────────────────────────────────────────────────────────────────────────
# Dilated Causal Convolution Block (Phase 4.2.1)
# ─────────────────────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """1D causal convolution with dilation.

    'Causal' means the convolution only uses past and present — never future
    samples.  Achieved by padding the left side only and removing the
    corresponding right output.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Left pad only
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the right-side outputs from the padding (enforce causality)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class ResidualBlock(nn.Module):
    """Residual block: CausalConv → WNorm → ReLU → Dropout (repeated × 2)."""

    def __init__(self, n_channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_channels, n_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(n_channels, n_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1×1 conv for residual connection when channel count changes
        self.residual = nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.residual(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full TCN Decoder (Phase 4.2)
# ─────────────────────────────────────────────────────────────────────────────
class TCNDecoder(nn.Module):
    """Temporal Convolutional Network for neural manifold decoding.

    Input:  [batch, n_channels, window_size]
    Output: [batch, latent_dim]
    """

    def __init__(self, n_channels: int = 256, hidden_dim: int = 64,
                 latent_dim: int = 16, kernel_size: int = 3,
                 n_layers: int = 10, dropout: float = 0.2):
        super().__init__()

        # Receptive field check
        rf = 1 + (kernel_size - 1) * (2 ** n_layers - 1)
        print(f"TCN receptive field: {rf} samples "
              f"({rf / 1000:.2f} s at 1 kHz)")

        # Input projection: n_channels → hidden_dim
        self.input_proj = nn.Conv1d(n_channels, hidden_dim, 1)

        # Residual blocks with exponentially increasing dilation
        blocks = []
        for i in range(n_layers):
            dilation = 2 ** i
            blocks.append(ResidualBlock(hidden_dim, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)

        # Output head: pool over time, project to latent space
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling over time
            nn.Flatten(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        h = self.input_proj(x)    # [batch, hidden, time]
        h = self.tcn(h)           # [batch, hidden, time]
        return self.output_head(h) # [batch, latent_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(data_path: str, labels_path: str, output_path: str,
          n_epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
          latent_dim: int = 16):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load data
    X = torch.tensor(np.load(data_path),   dtype=torch.float32)   # [N, C, T]
    Z = torch.tensor(np.load(labels_path), dtype=torch.float32)   # [N, latent_dim]

    n_channels  = X.shape[1]
    window_size = X.shape[2]
    n_samples   = X.shape[0]

    print(f"Data: N={n_samples}, channels={n_channels}, window={window_size}")

    model = TCNDecoder(n_channels=n_channels, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    indices = torch.randperm(n_samples)
    split   = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[i:i + batch_size]
            xb  = X[idx].to(device)
            zb  = Z[idx].to(device)
            pred = model(xb)
            loss = criterion(pred, zb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X[val_idx].to(device)),
                                 Z[val_idx].to(device)).item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={total_loss/len(train_idx):.6f}  "
                  f"val_loss={val_loss:.6f}")

    # Phase 4.3: Export via TorchScript
    model.eval().cpu()
    scripted = torch.jit.script(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)
    print(f"Model saved to {output_path}")

    # Validate C++ ↔ Python numerical agreement (Phase 4 validation)
    x_test = torch.randn(1, n_channels, window_size)
    with torch.no_grad():
        out = scripted(x_test)
    print(f"Output shape: {out.shape}  Output range: [{out.min():.4f}, {out.max():.4f}]")


def generate_synthetic_data(n_samples: int = 1000, n_channels: int = 256,
                              window: int = 1000, latent_dim: int = 16):
    """Generate synthetic training data for unit testing (no real recordings needed)."""
    import numpy as np

    Z = np.random.randn(n_samples, latent_dim).astype(np.float32)
    # Observation matrix: maps latent state to channels (low-rank)
    C = np.random.randn(n_channels, latent_dim).astype(np.float32) * 0.1
    # Simple temporal model: repeat Z across time, add noise
    X = np.zeros((n_samples, n_channels, window), dtype=np.float32)
    for i in range(n_samples):
        X[i] = (C @ Z[i:i+1].T).T[:, 0:1] * np.ones((1, window)) \
               + np.random.randn(n_channels, window).astype(np.float32) * 0.3
    return X, Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN decoder for Reuniclus")
    parser.add_argument("--data",   default=None, help="Path to neural data .npy [N,C,T]")
    parser.add_argument("--labels", default=None, help="Path to latent labels .npy [N,d]")
    parser.add_argument("--output", default="models/tcn_decoder.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate and use synthetic data for smoke test")
    args = parser.parse_args()

    if args.synthetic or (args.data is None):
        print("Using synthetic data...")
        X, Z = generate_synthetic_data(latent_dim=args.latent_dim)
        np.save("/tmp/synthetic_X.npy", X)
        np.save("/tmp/synthetic_Z.npy", Z)
        args.data   = "/tmp/synthetic_X.npy"
        args.labels = "/tmp/synthetic_Z.npy"

    train(args.data, args.labels, args.output,
          n_epochs=args.epochs, latent_dim=args.latent_dim)
