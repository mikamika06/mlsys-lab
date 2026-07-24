"""Deterministic fixture: per-sequence token counts for a mixed-length batch.

Run with:
    python3 tasks/rwb-attention-flops-packed-vs-padded/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(17)
    # a realistic mixed-length prefill batch: mostly short, a few long tails
    lens = rng.integers(8, 64, size=48)
    lens = np.concatenate([lens, rng.integers(256, 512, size=4)])
    rng.shuffle(lens)
    return lens.astype(np.int64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lens = build()
    np.save(OUT / "lens.npy", lens)
    print("wrote", lens.shape)
