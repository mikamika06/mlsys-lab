"""Deterministic fixture for rwb-streaming-softmax-running-max-sum.

A 37-element fp32 score vector (length not a multiple of the graded
chunk_size, so a ragged last chunk is exercised) with wide dynamic range,
including one very large and one very negative outlier -- exactly the
regime where a naive (non-online) softmax would overflow/underflow if
implemented directly in fp32.

Run with:

    python3 tasks/rwb-streaming-softmax-running-max-sum/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(21)
    n = 37
    x = rng.uniform(-50.0, 800.0, size=n).astype(np.float32)
    x[5] = 1000.0
    x[20] = -1000.0
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = build()
    np.save(OUT / "scores.npy", x)
    print("wrote", x.shape, x.dtype)
