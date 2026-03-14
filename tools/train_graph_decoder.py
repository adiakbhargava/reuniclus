"""
Phase 7.2.3 – Graph Neural Network Decoder Training (Python Side)
=================================================================
Trains a ConnectomeDecoder (GNN + TCN) and exports a TorchScript model.

Architecture:
  Spatial GNN layers (message-passing weighted by connectome adjacency)
  → Temporal TCN per node (Phase 4 TemporalBlock, dilation=4)
  → Latent projection head

Input:  [batch, channels, time]
Output: [batch, latent_dim]

Graph convolution (Phase 7 Appendix A):
  h_i^{(l+1)} = σ( Σ_{j ∈ N(i)} A_ij · W^{(l)} · h_j^{(l)} )

Usage:
  python tools/train_graph_decoder.py \
      --adj data/connectome_adj.npy \
      --data data/neural_sessions.npy \
      --labels data/latent_states.npy \
      --output models/graph_decoder.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Graph Convolution Layer (Phase 7.2.3)
# ─────────────────────────────────────────────────────────────────────────────
class GraphConvLayer(nn.Module):
    """Message-passing GNN layer constrained by connectome adjacency.

    h^{l+1} = σ( A_norm · H^{l} · W^{l} )

    where A_norm = D^{-1/2} A D^{-1/2} (symmetrically normalised adjacency).
    """

    def __init__(self, in_features: int, out_features: int,
                 adj: torch.Tensor):
        super().__init__()
        self.W   = nn.Linear(in_features, out_features, bias=True)
        # Normalised adjacency: A_norm = D^{-1/2} A D^{-1/2}
        degree   = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        d_inv_sq = degree.pow(-0.5)
        A_norm   = d_inv_sq * adj * d_inv_sq.T
        self.register_buffer("A_norm", A_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, features]
        # Message passing: aggregate neighbours weighted by adjacency
        h = torch.einsum("ij,bjf->bif", self.A_norm, x)  # Spatial aggregation
        return F.relu(self.W(h))


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Block (Phase 4.2 reused in Phase 7.2.3)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """Dilated causal 1D conv block applied per-channel in the GNN decoder."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 4):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=padding)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return F.relu(out[:, :, :-self.padding] if self.padding > 0 else out)


# ─────────────────────────────────────────────────────────────────────────────
# ConnectomeDecoder (Phase 7.2.3)
# ─────────────────────────────────────────────────────────────────────────────
class ConnectomeDecoder(nn.Module):
    """Connectome-constrained GNN + TCN decoder.

    Input:  [batch, channels, time]
    Output: [batch, latent_dim]
    """

    def __init__(self, adj_matrix: np.ndarray, n_channels: int,
                 latent_dim: int = 16):
        super().__init__()
        adj = torch.tensor(adj_matrix, dtype=torch.float32)
        # The adjacency is a non-learnable biological prior
        self.register_buffer("adj", nn.Parameter(adj, requires_grad=False))

        self.gnn1 = GraphConvLayer(1,  32, adj)
        self.gnn2 = GraphConvLayer(32, 64, adj)

        # Temporal convolution per-channel after spatial aggregation
        self.tcn = TemporalBlock(64 * n_channels, 64 * n_channels,
                                  kernel_size=3, dilation=4)

        self.head = nn.Linear(64 * n_channels, latent_dim)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        batch, C, T = x.shape

        # Reshape for GNN: treat each time step as one feature per channel
        # Apply GNN along the channel dimension at each time step
        x_t = x.permute(0, 2, 1).unsqueeze(-1)    # [batch, time, channels, 1]
        x_t = x_t.reshape(batch * T, C, 1)         # [batch*time, channels, 1]

        h = self.gnn1(x_t)                          # [batch*time, channels, 32]
        h = self.gnn2(h)                             # [batch*time, channels, 64]

        h = h.reshape(batch, T, C * 64)             # [batch, time, C*64]
        h = h.permute(0, 2, 1)                       # [batch, C*64, time]

        h = self.tcn(h)                              # Temporal convolution

        # Use last time step
        h = h[:, :, -1]                              # [batch, C*64]
        return self.head(h)                          # [batch, latent_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(adj_path: str, data_path: str, labels_path: str,
          output_path: str, n_epochs: int = 50, batch_size: int = 16,
          lr: float = 1e-3, latent_dim: int = 16):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    adj    = np.load(adj_path) if adj_path else None
    X      = torch.tensor(np.load(data_path),   dtype=torch.float32)
    Z      = torch.tensor(np.load(labels_path), dtype=torch.float32)
    n_ch   = X.shape[1]

    if adj is None:
        print("No adjacency matrix provided – using identity graph")
        adj = (np.eye(n_ch, k=1) + np.eye(n_ch, k=-1)).astype(np.float32)

    model = ConnectomeDecoder(adj, n_ch, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = X.shape[0]
    idx   = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[i:i + batch_size]
            loss = criterion(model(X[b].to(device)), Z[b].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X[va_idx].to(device)),
                                     Z[va_idx].to(device)).item()
            print(f"Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={total/len(tr_idx):.6f}  val={val_loss:.6f}")

    # Export TorchScript
    model.eval().cpu()
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(
            model, torch.randn(1, n_ch, X.shape[2]))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)
    print(f"Graph decoder saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj",    default=None,
                        help="Connectome adjacency matrix .npy [N×N]")
    parser.add_argument("--data",   default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default="models/graph_decoder.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic or args.data is None:
        from train_tcn import generate_synthetic_data
        X, Z = generate_synthetic_data(n_samples=200, n_channels=64,
                                        window=100, latent_dim=args.latent_dim)
        np.save("/tmp/synth_X.npy", X)
        np.save("/tmp/synth_Z.npy", Z)
        args.data   = "/tmp/synth_X.npy"
        args.labels = "/tmp/synth_Z.npy"

    train(args.adj, args.data, args.labels, args.output,
          n_epochs=args.epochs, latent_dim=args.latent_dim)
