"""Deterministic weight matrix and per-input-channel activation norms for
the Wanda per-row scale-invariance task.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

ROWS, COLS = 6, 24


def main() -> None:
    rng = np.random.default_rng(0)

    W = rng.standard_normal((ROWS, COLS)).astype(np.float64)
    col_norms = rng.uniform(0.1, 5.0, size=COLS).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "wanda_w.npy", W)
    np.save(out / "wanda_norms.npy", col_norms)


if __name__ == "__main__":
    main()
