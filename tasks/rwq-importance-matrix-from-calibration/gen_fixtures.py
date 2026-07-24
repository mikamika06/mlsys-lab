"""Deterministic fixture for rwq-importance-matrix-from-calibration.

Builds a synthetic calibration-activation matrix ``X`` of shape
``(n_tokens, n_channels)``: most channels have modest, per-channel-scaled
Gaussian activations, a handful of channels are "hot" (systematically much
larger, as real hidden-state channels used by down-projections often are),
and a few tokens carry extra activation spikes on random channels — the kind
of heavy-tailed calibration signal an imatrix run over real text produces.

Run with:
    python3 tasks/rwq-importance-matrix-from-calibration/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

N_TOKENS = 800
N_CHANNELS = 96

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(42)
    scale = rng.uniform(0.05, 1.0, size=N_CHANNELS)
    hot = rng.choice(N_CHANNELS, size=6, replace=False)
    scale[hot] *= rng.uniform(5.0, 12.0, size=6)

    X = rng.standard_normal((N_TOKENS, N_CHANNELS)) * scale

    spike_tokens = rng.choice(N_TOKENS, size=20, replace=False)
    spike_channels = rng.choice(N_CHANNELS, size=20)
    signs = np.where(rng.standard_normal(20) >= 0, 1.0, -1.0)
    X[spike_tokens, spike_channels] += rng.uniform(3.0, 8.0, size=20) * signs

    return X.astype(np.float32)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    X = build()
    np.save(OUT / "gguf_x.npy", X)
    print("wrote", X.shape)
