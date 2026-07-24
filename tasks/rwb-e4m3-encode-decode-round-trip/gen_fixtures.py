"""Deterministic fp32 probe fixture for the E4M3FN encode/decode task.

Builds the real 128-point nonnegative E4M3FN grid from its bit-pattern
decode (never hardcoded), then samples: exact grid points, round-to-
nearest-even tie midpoints, ordinary off-grid values, saturating values
past +-448, zeros, negatives, and broad random coverage.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def _decode_bits(code: np.ndarray) -> np.ndarray:
    code = np.asarray(code, dtype=np.uint8)
    sign = np.where((code & 0x80) != 0, -1.0, 1.0)
    e = ((code >> 3) & 0x0F).astype(np.int64)
    m = (code & 0x07).astype(np.int64)
    normal = sign * (1.0 + m / 8.0) * np.exp2((e - 7).astype(np.float64))
    subnormal = sign * (m / 8.0) * np.exp2(-6.0)
    val = np.where(e == 0, subnormal, normal)
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return val


def main() -> None:
    rng = np.random.default_rng(2026)

    nonneg_codes = np.arange(0, 127, dtype=np.uint8)  # excludes 0x7F (NaN)
    grid = _decode_bits(nonneg_codes)                 # ascending, grid[-1] == 448

    on_grid = grid[[0, 1, 2, 5, 20, 40, 60, 90, 110, 126]]

    tie_pairs = [(2, 3), (3, 4), (10, 11), (40, 41), (41, 42), (80, 81), (100, 101), (125, 126)]
    ties = np.array([(grid[a] + grid[b]) / 2.0 for a, b in tie_pairs])

    off_grid = grid[[8, 30, 55, 75, 95, 115]] * 1.07

    saturating = np.array([448.0, 448.1, 449.0, 500.0, 1000.0, 1e6])

    broad_random = rng.uniform(-500.0, 500.0, size=200)
    tiny_random = rng.uniform(-0.02, 0.02, size=100)  # subnormal-range magnitudes

    positive = np.concatenate([on_grid, ties, off_grid, saturating])
    values = np.concatenate(
        [[0.0, -0.0], positive, -positive, broad_random, tiny_random]
    )

    x = values.astype(np.float32)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "x.npy", x)


if __name__ == "__main__":
    main()
