"""
Train TCN encoder + classification head on BCI Competition IV Dataset 2a.
Evaluates per-subject (correct protocol for EEG).

Usage:
    python tools/train_tcn_eeg.py
    python tools/train_tcn_eeg.py --epochs 50 --subjects 1 2 3

Output:
    models/tcn_eeg.pt  — TorchScript model (22ch x 1001 → 4-class logits)
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# ── TCN building blocks ───────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.act(self.norm(self.conv(x)))


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        return x + self.drop(self.conv2(self.conv1(x)))


class TCNClassifier(nn.Module):
    """
    Input : (batch, n_channels, n_timepoints)
    Output: (batch, n_classes) — raw logits
    """
    def __init__(self, n_channels=22, n_classes=4, hidden=64):
        super().__init__()
        self.input_proj = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(hidden, kernel=3, dilation=1),
            TCNBlock(hidden, kernel=3, dilation=2),
            TCNBlock(hidden, kernel=3, dilation=4),
            TCNBlock(hidden, kernel=3, dilation=8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_subject(X, y, epochs, device):
    """5-fold CV on one subject, returns mean accuracy."""
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    n_cls  = len(le.classes_)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc)):
        X_tr = torch.tensor(X[tr], dtype=torch.float32)
        y_tr = torch.tensor(y_enc[tr], dtype=torch.long)
        X_te = torch.tensor(X[te], dtype=torch.float32)
        y_te = torch.tensor(y_enc[te], dtype=torch.long)

        ds  = TensorDataset(X_tr, y_tr)
        dl  = DataLoader(ds, batch_size=32, shuffle=True)

        model = TCNClassifier(n_channels=X.shape[1],
                              n_classes=n_cls).to(device)
        opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            sched.step()
            if (ep + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(X_tr.to(device)).argmax(1).cpu() == y_tr).float().mean().item()
                model.train()
                print(f"    fold {fold+1} epoch {ep+1:3d}/{epochs}  train_acc={tr_acc:.3f}", flush=True)

        model.eval()
        with torch.no_grad():
            preds = model(X_te.to(device)).argmax(1).cpu()
        acc = (preds == y_te).float().mean().item()
        fold_accs.append(acc)
        print(f"  fold {fold+1} test_acc={acc:.3f}", flush=True)

    return np.mean(fold_accs), model  # return last fold's model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=40)
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    parser.add_argument("--output",   default="models/tcn_eeg.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=32, tmin=0, tmax=4)
    dataset  = BNCI2014_001()

    subject_accs = []
    best_acc     = 0.0
    best_model   = None

    for subj in args.subjects:
        print(f"\n── Subject {subj} ──────────────────────────")
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subj])
        # X: (trials, 22, 1001)  y: string labels

        acc, model = train_subject(X, y, args.epochs, device)
        subject_accs.append(acc)
        print(f"  Accuracy : {acc:.3f}")

        if acc > best_acc:
            best_acc   = acc
            best_model = model

    accs = np.array(subject_accs)
    print(f"\n{'='*45}")
    print(f"TCN per-subject mean : {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"Best subject         : {accs.max():.3f}")
    print(f"Target               : ≥0.700")
    print(f"{'='*45}")

    # Save best model as TorchScript
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    best_model.eval().cpu()
    dummy = torch.zeros(1, 22, 1001)
    try:
        scripted = torch.jit.script(best_model.cpu())
    except Exception:
        scripted = torch.jit.trace(best_model.cpu(), dummy)
    scripted.save(args.output)
    print(f"Model saved → {args.output}")


if __name__ == "__main__":
    main()
