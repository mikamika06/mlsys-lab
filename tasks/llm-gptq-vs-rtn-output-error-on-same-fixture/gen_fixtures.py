"""Deterministic calibration fixture for the GPTQ-vs-RTN task.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(7)
    d_in, d_out, n_cal = 48, 32, 256

    # Correlated activations: a random mixing matrix makes X^T X strongly
    # non-diagonal, which is exactly the regime where GPTQ's error
    # compensation buys something over round-to-nearest.
    mix = rng.normal(size=(d_in, d_in)) / np.sqrt(d_in)
    X = (rng.normal(size=(n_cal, d_in)) @ mix).astype(np.float64)
    W = rng.normal(size=(d_out, d_in)).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "W.npy", W)
    np.save(out / "X.npy", X)


if __name__ == "__main__":
    main()
