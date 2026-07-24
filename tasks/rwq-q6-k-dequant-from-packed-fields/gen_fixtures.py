"""Deterministic fixture for rwq-q6-k-dequant-from-packed-fields.

A single GGML Q6_K super-block (QK_K = 256 elements): the packed 4-bit low
nibbles (`ql`), 2-bit high bits (`qh`), 16 signed 8-bit sub-block scales
(`scales`), and one super-block float scale (`d`). Every byte value is a
legal Q6_K bit pattern (the format has no reserved/invalid codes), so these
fields are drawn directly and deterministically rather than produced by
running a full encoder.

Run with:
    python3 tasks/rwq-q6-k-dequant-from-packed-fields/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(11)
    ql = rng.integers(0, 256, size=128, dtype=np.uint8)       # QK_K/2
    qh = rng.integers(0, 256, size=64, dtype=np.uint8)        # QK_K/4
    scales = rng.integers(-32, 32, size=16, dtype=np.int8)    # QK_K/16
    d = np.float64(rng.uniform(0.001, 0.05))
    return ql, qh, scales, d


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ql, qh, scales, d = build()
    np.save(OUT / "q6k_ql.npy", ql)
    np.save(OUT / "q6k_qh.npy", qh)
    np.save(OUT / "q6k_scales.npy", scales)
    np.save(OUT / "q6k_d.npy", np.array(d, dtype=np.float64))
    print("wrote ql", ql.shape, "qh", qh.shape, "scales", scales.shape, "d", d)
