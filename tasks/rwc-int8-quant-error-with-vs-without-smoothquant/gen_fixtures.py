"""Outlier-heavy activations (a few input channels 10-30x the typical
magnitude, as real LLM activations exhibit) paired with a comparatively
well-behaved weight matrix -- exactly the asymmetry SmoothQuant is
designed to migrate away from activations and onto weights.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(109)
    n, d_in, d_out = 64, 20, 16

    X = rng.standard_normal((n, d_in)) * 0.5
    outlier_channels = rng.choice(d_in, size=3, replace=False)
    X[:, outlier_channels] *= rng.uniform(10.0, 30.0, size=3)

    W = rng.standard_normal((d_out, d_in)) * 0.3

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "W.npy", W)


if __name__ == "__main__":
    main()
