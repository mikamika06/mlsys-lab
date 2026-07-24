"""Deterministic raw NVFP4 storage arrays (E4M3 block-scale codes, E2M1
element codes, one fp32 global scale) for the reconstruction task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(9)

    n_blocks = 40
    block = 16

    # E4M3 byte codes for the per-block scale magnitudes: sign bit fixed to
    # 0 (block scales are positive magnitudes), exponent/mantissa sampled
    # broadly but excluding the reserved NaN code (E=15, M=7).
    exp_bits = rng.integers(0, 16, size=n_blocks)
    man_bits = rng.integers(0, 8, size=n_blocks)
    is_nan = (exp_bits == 15) & (man_bits == 7)
    man_bits = np.where(is_nan, 0, man_bits)  # nudge the rare NaN draw off the reserved code
    e4m3_block_codes = ((exp_bits << 3) | man_bits).astype(np.uint8)

    # E2M1 4-bit codes: every one of the 16 signed codes is a legal draw.
    e2m1_codes = rng.integers(0, 16, size=(n_blocks, block)).astype(np.uint8)

    global_scale = np.array(0.017578125, dtype=np.float32)  # an exact power-of-two-ish fp32 constant

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "e4m3_block_codes.npy", e4m3_block_codes)
    np.save(out / "e2m1_codes.npy", e2m1_codes)
    np.save(out / "global_scale.npy", global_scale)


if __name__ == "__main__":
    main()
