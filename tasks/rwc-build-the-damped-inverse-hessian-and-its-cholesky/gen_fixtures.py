"""Calibration activations with correlated features (a mixing matrix
makes X^T X strongly non-diagonal), the regime GPTQ's Hessian-based
update is actually built for.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(103)
    n_cal, d_in = 200, 24

    mix = rng.standard_normal((d_in, d_in)) / np.sqrt(d_in)
    X = (rng.standard_normal((n_cal, d_in)) @ mix).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "X.npy", X)


if __name__ == "__main__":
    main()
