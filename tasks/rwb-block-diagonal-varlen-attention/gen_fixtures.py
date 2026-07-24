"""Deterministic packed varlen-attention fixture: a realistic skewed batch
(one long prefill-like sequence among several short decode-like ones).

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(31)
    d = 8

    # Skewed batch: a few short sequences, one long one, one more short --
    # mirrors a real continuous-batching mix of prefill + decode requests.
    lengths = [3, 2, 5, 47, 1, 4, 9, 2]
    cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    N = int(cu_seqlens[-1])

    q = rng.standard_normal((N, d)).astype(np.float64)
    k = rng.standard_normal((N, d)).astype(np.float64)
    v = rng.standard_normal((N, d)).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "q.npy", q)
    np.save(out / "k.npy", k)
    np.save(out / "v.npy", v)
    np.save(out / "cu_seqlens.npy", cu_seqlens)


if __name__ == "__main__":
    main()
