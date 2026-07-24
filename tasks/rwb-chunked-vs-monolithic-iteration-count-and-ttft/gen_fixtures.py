"""Deterministic realistic-skew prefill workload: several short prompts
plus a couple of long ones, and a token budget C that does not evenly
divide most prompt lengths.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    prompt_lens = np.array([12, 5, 3, 200, 7, 40, 2, 9, 130, 6, 4, 15], dtype=np.int64)
    budget = np.array(64, dtype=np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "prompt_lens.npy", prompt_lens)
    np.save(out / "budget.npy", budget)


if __name__ == "__main__":
    main()
