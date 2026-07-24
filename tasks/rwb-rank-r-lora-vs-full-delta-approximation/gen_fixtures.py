"""A full weight-delta matrix (as if from full fine-tuning) with a fast
singular-value decay -- typical of real fine-tuning deltas, which is
exactly why low-rank LoRA approximates them well -- plus an arbitrary
(non-optimal) rank-r factor pair A0, B0 for comparison.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(79)
    d_out, d_in, true_rank = 40, 32, 6

    U = rng.standard_normal((d_out, true_rank))
    V = rng.standard_normal((d_in, true_rank))
    decay = np.array([10.0, 6.0, 3.5, 2.0, 1.1, 0.6])
    W = (U * decay) @ V.T
    W += rng.standard_normal((d_out, d_in)) * 0.05  # small full-rank noise floor

    r = 4
    A0 = rng.standard_normal((d_out, r))
    B0 = rng.standard_normal((r, d_in))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "W.npy", W)
    np.save(out / "A0.npy", A0)
    np.save(out / "B0.npy", B0)


if __name__ == "__main__":
    main()
