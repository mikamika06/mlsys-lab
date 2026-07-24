"""Synthetic but realistic KV-cache tensors: mostly small-magnitude
values with a handful of outlier channels, similar to real transformer
K/V activation statistics.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(71)
    seq_len, d = 96, 32

    def make_tensor():
        x = rng.standard_normal((seq_len, d)).astype(np.float64) * 0.5
        # a few outlier channels/columns with much larger magnitude
        outlier_cols = rng.choice(d, size=3, replace=False)
        x[:, outlier_cols] *= rng.uniform(8.0, 20.0, size=3)
        return x

    K = make_tensor()
    V = make_tensor()

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "k.npy", K)
    np.save(out / "v.npy", V)


if __name__ == "__main__":
    main()
