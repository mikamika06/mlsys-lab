"""Deterministic weight matrix for the qint4 per-axis vs per-group task.
Shape (8, 128) with group_size=32 (4 groups per row). Each group within a
row is generated with its own randomly chosen magnitude scale, so a single
per-row scale is a poor fit for every group while a per-group scale fits
each segment tightly -- this is what makes per-group quantization strictly
lower error than per-axis on this fixture.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

ROWS, COLS = 8, 128
GROUP_SIZE = 32


def main() -> None:
    rng = np.random.default_rng(0)
    ng = COLS // GROUP_SIZE
    W = np.empty((ROWS, COLS), dtype=np.float64)

    for r in range(ROWS):
        for g in range(ng):
            scale = float(10.0 ** rng.uniform(-2.0, 1.0))  # 0.01 .. 10
            seg = rng.standard_normal(GROUP_SIZE) * scale
            W[r, g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = seg

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "qnt_w.npy", W)


if __name__ == "__main__":
    main()
