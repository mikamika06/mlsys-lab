"""Deterministic fixture of prompt lengths for the chunked-prefill task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    hand_picked = [1, 2, 511, 512, 513, 1024, 1025, 4096]
    random_lens = rng.integers(1, 8000, size=60)

    prompt_lens = np.concatenate([hand_picked, random_lens]).astype(np.int64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "prompt_lens.npy", prompt_lens)


if __name__ == "__main__":
    main()
