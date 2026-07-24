"""Deterministic fixtures for rwb-reconstruct-global-attention-weights-from-per-chunk-lse.

Simulates what a chunked/ring-attention worker would have on hand for one
query row, split across C=6 key/value chunks of chunk_size=4 tokens each
(24 keys total): the raw per-token scores, each chunk's own log-sum-exp
(a single scalar reduction, as would be exchanged between workers), and
each chunk's local unnormalized partial output (sum_j exp(score_j - m_c) * V_j).

Run with:

    python3 tasks/rwb-reconstruct-global-attention-weights-from-per-chunk-lse/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

C, CHUNK_SIZE, D = 6, 4, 4


def build():
    rng = np.random.default_rng(9)
    chunk_scores = rng.normal(0.0, 4.0, size=(C, CHUNK_SIZE))
    V = rng.normal(0.0, 1.0, size=(C, CHUNK_SIZE, D))

    m = chunk_scores.max(axis=1, keepdims=True)
    chunk_lse = m[:, 0] + np.log(np.exp(chunk_scores - m).sum(axis=1))
    chunk_partial_out = (np.exp(chunk_scores - m)[:, :, None] * V).sum(axis=1)

    return (
        chunk_scores.astype(np.float64),
        chunk_lse.astype(np.float64),
        chunk_partial_out.astype(np.float64),
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chunk_scores, chunk_lse, chunk_partial_out = build()
    np.save(OUT / "chunk_scores.npy", chunk_scores)
    np.save(OUT / "chunk_lse.npy", chunk_lse)
    np.save(OUT / "chunk_partial_out.npy", chunk_partial_out)
    print("wrote", chunk_scores.shape, chunk_lse.shape, chunk_partial_out.shape)
