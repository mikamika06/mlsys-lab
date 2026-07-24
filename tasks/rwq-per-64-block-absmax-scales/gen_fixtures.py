"""A deterministic 'real Linear weight'-like matrix for the NF4-style
per-64-block absmax scale task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)

    out_features, in_features = 96, 256  # 24576 elements, divisible by 64
    # Per-output-row scale variation, like a real trained Linear layer where
    # different output channels have different weight magnitudes.
    row_scale = rng.uniform(0.05, 3.0, size=(out_features, 1))
    W = rng.standard_normal((out_features, in_features)) * row_scale

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "nf4_w.npy", W)


if __name__ == "__main__":
    main()
