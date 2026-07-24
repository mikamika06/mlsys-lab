"""Deterministic fixture for rwq-nearest-rounding-baseline-v-0.

AutoRound-style layer weight tensor: values shaped like a real LLM linear
layer's initialization (small Gaussian noise around zero, per-output-row
scale variation), which is what "ar_w.npy" (AutoRound weights) stands for.

Run with:
    python3 tasks/rwq-nearest-rounding-baseline-v-0/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(7)
    rows, cols = 96, 256
    row_scale = rng.uniform(0.01, 0.05, size=(rows, 1))
    W = rng.standard_normal((rows, cols)) * row_scale
    return W.astype(np.float32)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    W = build()
    np.save(OUT / "ar_w.npy", W)
    print("wrote", W.shape)
