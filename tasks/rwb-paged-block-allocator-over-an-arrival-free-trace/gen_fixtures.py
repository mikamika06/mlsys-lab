"""Deterministic fixture for rwb-paged-block-allocator-over-an-arrival-free-trace.

A 25-request arrival/departure trace: each request i arrives at
arrive_t[i], holds its allocation until depart_t[i], and needs seq_len[i]
tokens of KV cache while alive. Arrivals overlap heavily (many requests
alive concurrently), so the pool genuinely fills up and later arrivals can
be rejected, and later departures free capacity for still-later arrivals.

Run with:

    python3 tasks/rwb-paged-block-allocator-over-an-arrival-free-trace/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

MAX_LEN = 64  # the worst-case context length used by the contiguous comparison


def build():
    rng = np.random.default_rng(42)
    n = 25
    arrive_t = np.sort(rng.integers(0, 100, size=n))
    durations = rng.integers(5, 40, size=n)
    depart_t = arrive_t + durations
    seq_len = rng.integers(1, MAX_LEN + 1, size=n)
    return arrive_t.astype(np.int64), depart_t.astype(np.int64), seq_len.astype(np.int64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    arrive_t, depart_t, seq_len = build()
    np.save(OUT / "arrive_t.npy", arrive_t)
    np.save(OUT / "depart_t.npy", depart_t)
    np.save(OUT / "seq_len.npy", seq_len)
    print("wrote", arrive_t.shape, depart_t.shape, seq_len.shape)
