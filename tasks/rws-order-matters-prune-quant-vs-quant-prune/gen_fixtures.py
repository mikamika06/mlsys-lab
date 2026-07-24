"""Deterministic weight matrix and calibration activations for the
prune-then-quant vs quant-then-prune ordering task.

Each 16-column group has one "loud but unimportant" column: a large-magnitude
weight value paired with a tiny-variance activation column, so its Wanda
score is low and it reliably gets pruned -- yet its raw magnitude would
otherwise dominate a full-range quantization scale. This is exactly the
situation where the two orderings diverge.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

ROWS = 6
GROUP_SIZE = 16
GROUPS_PER_ROW = 2
COLS = GROUP_SIZE * GROUPS_PER_ROW
N_SAMPLES = 30


def main() -> None:
    rng = np.random.default_rng(0)

    X = np.empty((N_SAMPLES, COLS), dtype=np.float64)
    outlier_col = {}
    for g in range(GROUPS_PER_ROW):
        cols = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        x_std = np.full(GROUP_SIZE, 1.0)
        k = int(rng.integers(0, GROUP_SIZE))
        outlier_col[g] = k
        x_std[k] = 0.002  # tiny-variance "unimportant" input channel
        X[:, cols] = rng.standard_normal((N_SAMPLES, GROUP_SIZE)) * x_std[None, :]

    W = np.empty((ROWS, COLS), dtype=np.float64)
    for r in range(ROWS):
        for g in range(GROUPS_PER_ROW):
            cols = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
            w_block = rng.standard_normal(GROUP_SIZE) * 0.05
            k = outlier_col[g]
            mag = float(rng.uniform(15.0, 30.0)) * 0.05
            sign = 1.0 if rng.standard_normal() >= 0 else -1.0
            w_block[k] = mag * sign
            W[r, cols] = w_block

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "opw_w.npy", W)
    np.save(out / "opw_x.npy", X)


if __name__ == "__main__":
    main()
