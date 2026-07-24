"""Deterministic fixture for one GQA layer's K/V/Q (fp32).

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    seq_len = 128
    n_kv_heads = 8
    group_size = 4
    n_q_heads = n_kv_heads * group_size
    head_dim = 64

    k = rng.normal(size=(seq_len, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.normal(size=(seq_len, n_kv_heads, head_dim)).astype(np.float32)
    q = rng.normal(size=(n_q_heads, head_dim)).astype(np.float32)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "k.npy", k)
    np.save(out / "v.npy", v)
    np.save(out / "q.npy", q)


if __name__ == "__main__":
    main()
