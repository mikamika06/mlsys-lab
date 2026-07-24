"""Deterministic fixture: per-request total (prompt+generation) length,
grouped into fixed-size static batches via a parallel batch-id array.

Run with:
    python3 tasks/rwb-measure-padding-waste-of-static-batching/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(23)
    num_batches = 8
    lens_list = []
    batch_ids_list = []
    for b in range(num_batches):
        batch_size = int(rng.integers(4, 17))
        lens = rng.integers(16, 1024, size=batch_size)
        lens_list.append(lens)
        batch_ids_list.append(np.full(batch_size, b, dtype=np.int64))
    lens = np.concatenate(lens_list).astype(np.int64)
    batch_ids = np.concatenate(batch_ids_list).astype(np.int64)
    return lens, batch_ids


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lens, batch_ids = build()
    np.save(OUT / "lens.npy", lens)
    np.save(OUT / "batch_ids.npy", batch_ids)
    print("wrote", lens.shape, batch_ids.shape)
