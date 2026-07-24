"""Deterministic fixtures for rwc-awq-vs-plain-rtn-4-bit-on-a-salient-heavy-weight.

A small linear layer (W: 16x32, X: 64x32) where three input channels are
made strongly salient (120x the typical activation magnitude) -- exactly
the regime AWQ's per-channel scaling is designed to protect: plain RTN
quantization of W corrupts those channels' contribution badly, since the
same INT4 grid has to cover both the salient and ordinary channels.

Run with:

    python3 tasks/rwc-awq-vs-plain-rtn-4-bit-on-a-salient-heavy-weight/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

SALIENT_CHANNELS = (2, 9, 20)
SALIENT_SCALE = 120.0


def build():
    rng = np.random.default_rng(13)
    out_dim, in_dim, batch = 16, 32, 64
    W = rng.normal(0.0, 1.0, size=(out_dim, in_dim))
    X = rng.normal(0.0, 1.0, size=(batch, in_dim))
    X[:, list(SALIENT_CHANNELS)] *= SALIENT_SCALE
    return W.astype(np.float64), X.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    W, X = build()
    np.save(OUT / "W.npy", W)
    np.save(OUT / "X.npy", X)
    print("wrote", W.shape, X.shape)
