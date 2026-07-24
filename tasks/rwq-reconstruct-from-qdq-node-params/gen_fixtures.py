"""Deterministic uint8 code tensor with a per-axis-0 scale/zero-point pair,
mimicking an ONNX QDQ (DequantizeLinear) node's stored parameters.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)

    rows, cols = 6, 10
    q = rng.integers(0, 256, size=(rows, cols)).astype(np.uint8)
    scale = rng.uniform(0.001, 0.5, size=rows).astype(np.float64)
    zero_point = rng.integers(0, 256, size=rows).astype(np.uint8)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "qdq_q.npy", q)
    np.save(out / "qdq_scale.npy", scale)
    np.save(out / "qdq_zp.npy", zero_point)


if __name__ == "__main__":
    main()
