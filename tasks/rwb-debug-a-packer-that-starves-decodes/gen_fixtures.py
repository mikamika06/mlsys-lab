"""Deterministic fixture of scheduler-step snapshots (token_budget,
num_running, prefill_remaining) for the token-budget packer task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    rows = []

    # Hand-picked edge cases.
    rows += [
        (0, 0, 0),      # nothing to do
        (0, 5, 10),     # zero budget, requests waiting
        (10, 0, 0),     # idle: budget unused
        (10, 0, 100),   # no running seqs: prefill can take the whole budget
        (10, 7, 0),     # no prefill: budget goes entirely to decode
        (5, 5, 0),      # exact fit, no prefill
        (5, 8, 0),      # budget can't even cover all decodes
        (16, 4, 64),    # classic starvation trap: prefill alone eats the budget
        (16, 16, 64),   # decode alone eats the whole budget, nothing left for prefill
        (1, 1, 1),      # smallest non-trivial contention
    ]

    # Broad random coverage, including plenty of starvation-triggering cases
    # (prefill_remaining >= token_budget while num_running > 0).
    for _ in range(80):
        token_budget = int(rng.integers(0, 65))
        num_running = int(rng.integers(0, 33))
        prefill_remaining = int(rng.integers(0, 129))
        rows.append((token_budget, num_running, prefill_remaining))

    state = np.array(rows, dtype=np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "state.npy", state)


if __name__ == "__main__":
    main()
