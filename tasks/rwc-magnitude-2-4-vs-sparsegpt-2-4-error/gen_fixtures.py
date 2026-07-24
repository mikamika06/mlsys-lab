"""Deterministic fixtures for rwc-magnitude-2-4-vs-sparsegpt-2-4-error.

A small linear layer (W: 8x16) with correlated, salient-channel-heavy
calibration activations (X: 64x16) -- the regime where Hessian-aware
SparseGPT pruning has real leverage over magnitude-only pruning: naive
magnitude pruning judges each weight only by its own size, so it happily
prunes a small weight sitting on a highly-activated (and/or highly
correlated) input channel, while SparseGPT's Hessian-based saliency score
and inverse-Hessian compensation account for exactly that.

Run with:

    python3 tasks/rwc-magnitude-2-4-vs-sparsegpt-2-4-error/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(7)
    out_dim, in_dim, s = 8, 16, 64

    W = rng.normal(0.0, 1.0, size=(out_dim, in_dim))

    # Correlated activations (mixing matrix) plus a few salient channels.
    A = rng.normal(0.0, 1.0, size=(in_dim, in_dim)) * 0.3 + np.eye(in_dim)
    Z = rng.normal(0.0, 1.0, size=(s, in_dim))
    X = Z @ A
    salient = rng.choice(in_dim, size=3, replace=False)
    X[:, salient] *= 15.0

    return W.astype(np.float64), X.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    W, X = build()
    np.save(OUT / "W.npy", W)
    np.save(OUT / "X.npy", X)
    print("wrote", W.shape, X.shape)
