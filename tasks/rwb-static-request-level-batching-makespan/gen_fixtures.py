"""Deterministic fixture: per-request decode-iteration counts, in queue
(arrival) order.

Run with:
    python3 tasks/rwb-static-request-level-batching-makespan/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(31)
    return rng.integers(5, 200, size=23).astype(np.int64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    output_lens = build()
    np.save(OUT / "output_lens.npy", output_lens)
    print("wrote", output_lens.shape)
