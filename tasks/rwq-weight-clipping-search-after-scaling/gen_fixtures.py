"""Deterministic weight matrix for the AWQ-style clip-ratio search task.
Shape (10, 64), group_size=32 (2 groups per row). Roughly half the groups
get a single injected outlier element several times larger than the rest of
the group -- exactly the situation where searching a clip ratio < 1.0 lowers
reconstruction MSE (a slightly clipped outlier costs less than the extra
quantization noise a too-wide scale forces onto the rest of the group).

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

ROWS = 10
GROUP_SIZE = 32
GROUPS_PER_ROW = 2
COLS = GROUP_SIZE * GROUPS_PER_ROW


def main() -> None:
    rng = np.random.default_rng(0)
    W = np.empty((ROWS, COLS), dtype=np.float64)

    for r in range(ROWS):
        for g in range(GROUPS_PER_ROW):
            seg = rng.standard_normal(GROUP_SIZE) * 0.02
            if (r + g) % 2 == 0:
                k = int(rng.integers(0, GROUP_SIZE))
                mag = float(rng.uniform(6.0, 15.0)) * 0.02
                sign = 1.0 if rng.standard_normal() >= 0 else -1.0
                seg[k] = mag * sign
            W[r, g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = seg

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "clip_w.npy", W)


if __name__ == "__main__":
    main()
