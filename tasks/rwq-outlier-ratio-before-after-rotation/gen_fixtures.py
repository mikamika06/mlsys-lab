"""Deterministic fixture for rwq-outlier-ratio-before-after-rotation.

Synthetic calibration activations shaped like a real transformer hidden
state: most channels carry modest Gaussian noise, but a handful of
"systematic outlier channels" (the kind repeatedly reported in LLM
activations, e.g. in OPT/LLaMA-family hidden states) are 15-40x larger for
every token. This is exactly the pattern the Hadamard rotation used by
QuaRot/SpinQuant-style quantizers is designed to flatten.

Run with:
    python3 tasks/rwq-outlier-ratio-before-after-rotation/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

N_TOKENS = 512
D = 64  # must be a power of two (Hadamard requirement)


def build():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N_TOKENS, D)) * 0.3
    outlier_channels = rng.choice(D, size=4, replace=False)
    X[:, outlier_channels] *= rng.uniform(15.0, 40.0, size=4)
    return X.astype(np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    X = build()
    np.save(OUT / "rot_x.npy", X)
    print("wrote", X.shape)
