"""Deterministic 1D array for the MXFP4 E8M0 block-scale task: 32-element
blocks covering an all-zero block, hand-picked amax values landing near
power-of-two boundaries of amax/6, and log-uniform random-magnitude blocks.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

BLOCK = 32


def _block(amax: float, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """A block whose largest-magnitude element is exactly `amax`."""
    b = rng.uniform(-amax, amax, size=block_size)
    b[0] = amax  # guarantee the exact amax value is present
    return b


def main() -> None:
    rng = np.random.default_rng(0)

    blocks = [np.zeros(BLOCK)]  # all-zero block: amax == 0 edge case

    # Hand-picked amax values: exact multiples of 6 * 2^k (clean E8M0 boundary
    # values) plus values that sit strictly inside a bracket.
    hand_amax = [6.0, 12.0, 3.0, 1.5, 0.75, 96.0, 6.0 * 2**5, 0.006 * 6, 9.0, 5.999]
    for amax in hand_amax:
        blocks.append(_block(amax, BLOCK, rng))

    # Log-uniform random-magnitude blocks.
    for _ in range(40):
        amax = float(10.0 ** rng.uniform(-6, 4))
        blocks.append(_block(amax, BLOCK, rng))

    x = np.concatenate(blocks).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "mx_w.npy", x)


if __name__ == "__main__":
    main()
