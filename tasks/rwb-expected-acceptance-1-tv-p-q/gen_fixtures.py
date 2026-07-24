"""Deterministic fixtures for rwb-expected-acceptance-1-tv-p-q.

`p` (target distribution) and `q` (draft distribution) over a small
vocabulary, built as softmax(logits) so they're genuine, non-trivial
probability vectors -- `q` is a noisy perturbation of `p`'s logits, as a
speculative-decoding draft model would produce.

Run with:

    python3 tasks/rwb-expected-acceptance-1-tv-p-q/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def build():
    rng = np.random.default_rng(7)
    logits_p = rng.normal(0.0, 1.5, size=10)
    logits_q = logits_p + rng.normal(0.0, 1.0, size=10)
    p = _softmax(logits_p).astype(np.float64)
    q = _softmax(logits_q).astype(np.float64)
    return p, q


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p, q = build()
    np.save(OUT / "p.npy", p)
    np.save(OUT / "q.npy", q)
    print("wrote", p.shape, q.shape)
