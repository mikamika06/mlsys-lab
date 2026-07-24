"""Deterministic request-length distribution fixture.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    # A realistic mix: mostly short-to-medium prompts, a handful of long
    # ones near the configured max context length -- enough spread that
    # paged packing clearly outperforms worst-case contiguous reservation.
    n = 60
    short = rng.integers(8, 120, size=40)
    medium = rng.integers(120, 350, size=15)
    long = rng.integers(350, 512, size=5)
    seqlens = np.concatenate([short, medium, long]).astype(np.int64)
    rng.shuffle(seqlens)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "seqlens.npy", seqlens)


if __name__ == "__main__":
    main()
