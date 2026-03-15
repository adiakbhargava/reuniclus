"""
Phase 1 validation: run BCI Competition IV Dataset 2a through
the Reuniclus TCN decoder and report 4-class motor imagery accuracy.
"""
import numpy as np
from moabb.datasets import BNCI2014_001        # BCI Comp IV 2a
from moabb.paradigms import MotorImagery
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import torch, os, sys

# ── Load dataset ──────────────────────────────────────────────
paradigm  = MotorImagery(n_classes=4, fmin=8, fmax=32, tmin=0, tmax=4)
dataset   = BNCI2014_001()
X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1,2,3,4,5,6,7,8,9])

print(f"Data shape : {X.shape}")   # (trials, channels, timepoints)
print(f"Classes    : {np.unique(y)}")
print(f"Trials     : {len(y)}")

# ── Baseline: CSP + LDA per-subject (correct evaluation) ─────
# EEG is subject-specific — cross-subject evaluation is invalid for CSP.
# Must train and test within each subject separately.
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold

subject_scores = []
for subj in range(1, 10):
    Xs, ys, _ = paradigm.get_data(dataset=dataset, subjects=[subj])
    csp = CSP(n_components=8, reg=None, log=True)
    lda = LinearDiscriminantAnalysis()
    pipe = Pipeline([("csp", csp), ("lda", lda)])
    s = cross_val_score(pipe, Xs, ys, cv=5, scoring="accuracy", n_jobs=1)
    subject_scores.append(s.mean())
    print(f"  Subject {subj}: {s.mean():.3f}")

import numpy as np
subject_scores = np.array(subject_scores)
print(f"\nCSP+LDA per-subject: {subject_scores.mean():.3f} ± {subject_scores.std():.3f}")

# ── Load TCN decoder ──────────────────────────────────────────
model_path = "models/tcn_decoder.pt"
if os.path.exists(model_path):
    model = torch.jit.load(model_path)
    model.eval()
    # Resize to match model input: (trials, 64, 1000)
    # Dataset is 22ch × 1001 pts — resample to 64ch×1000 via interpolation
    from scipy.interpolate import interp1d
    X_re = np.zeros((X.shape[0], 64, 1000))
    for i in range(X.shape[0]):
        # channel interpolation 22→64
        ch_interp = interp1d(np.linspace(0,1,22), X[i], axis=0)
        X_re[i]   = ch_interp(np.linspace(0,1,64))[:, :1000]

    # Run inference in batches to avoid OOM
    batch_size = 64
    latents = []
    with torch.no_grad():
        for i in range(0, len(X_re), batch_size):
            batch = torch.tensor(X_re[i:i+batch_size], dtype=torch.float32)
            latents.append(model(batch).numpy())
    latents = np.concatenate(latents, axis=0)  # (trials, latent_dim)

    from sklearn.linear_model import LogisticRegression
    clf    = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, latents, y, cv=5, scoring="accuracy")
    print(f"TCN latent + LR   : {scores.mean():.3f} ± {scores.std():.3f}")
else:
    print(f"No TCN model found at {model_path} — skipping TCN eval")

print("\nTarget for funding application: ≥0.700")
