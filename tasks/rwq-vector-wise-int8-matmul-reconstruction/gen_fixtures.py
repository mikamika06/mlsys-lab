"""Deterministic 'no strong outliers' activation/weight fixtures for the
vector-wise int8 matmul reconstruction task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(11)

    n, d, m = 32, 64, 24

    # Mild, bounded magnitude variation row-to-row / col-to-col (no LLM.int8()
    # -style emergent outlier features) so per-vector int8 quantization noise
    # stays small and uniform across rows/columns.
    X = rng.standard_normal((n, d)) * rng.uniform(0.5, 2.0, size=(n, 1))
    W = rng.standard_normal((d, m)) * rng.uniform(0.5, 2.0, size=(1, m))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "int8_x.npy", X)
    np.save(out / "int8_w.npy", W)


if __name__ == "__main__":
    main()
