"""Deterministic 1D weight vector for the NVFP4 two-level scaling task.
16-element blocks with amax values spread around a common baseline (so the
per-16 block scale, after being folded through the per-tensor scale, stays
well inside the E4M3 representable range), plus one all-zero block to
exercise the amax == 0 edge case.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

BLOCK = 16
BASE_AMAX = 0.03


def _block(amax: float, block_size: int, rng: np.random.Generator) -> np.ndarray:
    b = rng.uniform(-amax, amax, size=block_size)
    b[0] = amax  # guarantee the exact amax value is present in the block
    return b


def main() -> None:
    rng = np.random.default_rng(0)

    blocks = [np.zeros(BLOCK)]  # all-zero block edge case

    for _ in range(50):
        factor = float(10.0 ** rng.uniform(-0.6, 0.6))  # ~0.25x .. 4x baseline
        blocks.append(_block(BASE_AMAX * factor, BLOCK, rng))

    # A couple of wider-spread blocks (still within safe underflow margin).
    blocks.append(_block(BASE_AMAX * 5.0, BLOCK, rng))
    blocks.append(_block(BASE_AMAX * 0.08, BLOCK, rng))

    w = np.concatenate(blocks).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "nv_w.npy", w)


if __name__ == "__main__":
    main()
