"""Deterministic fixture for the speculative-decoding modified rejection
sampling accept step: per-position target/draft distributions, drafted
token ids, and a fixed uniform RNG stream.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(2026)

    T = 25  # number of speculative positions
    V = 12  # vocab size

    p_logits = rng.normal(scale=1.5, size=(T, V))
    q_logits = rng.normal(scale=1.5, size=(T, V))
    p = _softmax(p_logits)
    q = _softmax(q_logits)

    # Drafted token ids: one categorical sample from q per position, using
    # a separate rng stream (independent of the graded uniform stream).
    draft_token_ids = np.array(
        [rng.choice(V, p=q[t]) for t in range(T)], dtype=np.int64
    )

    # Fixed uniform stream, worst case 2 draws per position (accept-check
    # + residual resample), consumed sequentially and stateful across
    # positions.
    u_stream = rng.random(2 * T)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "p.npy", p)
    np.save(out / "q.npy", q)
    np.save(out / "draft_token_ids.npy", draft_token_ids)
    np.save(out / "u_stream.npy", u_stream)


if __name__ == "__main__":
    main()
