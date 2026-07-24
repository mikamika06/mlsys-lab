"""Deterministic fixture for rwq-top-1-salient-channels.

AWQ-style calibration activations: most channels have modest average
magnitude, a handful of channels are systematically much larger (the
"salient" channels AWQ protects when quantizing the corresponding weight
columns).

Run with:
    python3 tasks/rwq-top-1-salient-channels/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

N_TOKENS = 600
N_CHANNELS = 200  # top-1% -> ceil(0.01 * 200) = 2 salient channels


def build():
    rng = np.random.default_rng(5)
    base_scale = rng.uniform(0.05, 0.4, size=N_CHANNELS)
    X = rng.standard_normal((N_TOKENS, N_CHANNELS)) * base_scale

    salient = rng.choice(N_CHANNELS, size=2, replace=False)
    X[:, salient] *= rng.uniform(8.0, 20.0, size=2)

    return X.astype(np.float32)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    X = build()
    np.save(OUT / "awq_x.npy", X)
    print("wrote", X.shape)
