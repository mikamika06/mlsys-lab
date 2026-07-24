"""Deterministic fp32 array covering every E4M3 regime: exact zero,
underflow-to-zero, subnormal, normal, and overflow-clamped, plus a large
batch of log-uniform random magnitudes (with random signs) sprinkled across
the same ranges so the classifier can't just memorize the hand-picked
boundary values.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

MIN_SUBNORMAL = 2.0 ** -9   # 0.001953125
MIN_NORMAL = 2.0 ** -6      # 0.015625
MAX_NORMAL = 448.0


def main() -> None:
    rng = np.random.default_rng(0)

    hand_picked = np.array([
        0.0, -0.0,
        1e-10, -1e-10,                                   # deep underflow
        MIN_SUBNORMAL * 0.4, -MIN_SUBNORMAL * 0.4,        # underflow, close to boundary
        MIN_SUBNORMAL * 0.999,                            # just under min-subnormal
        MIN_SUBNORMAL, -MIN_SUBNORMAL,                    # exactly min-subnormal -> subnormal
        MIN_SUBNORMAL * 3.5,                              # mid-subnormal
        MIN_NORMAL * 0.999,                               # just under min-normal -> subnormal
        MIN_NORMAL, -MIN_NORMAL,                          # exactly min-normal -> normal
        1.0, -1.0, 100.0,
        MAX_NORMAL, -MAX_NORMAL,                          # exactly max -> normal
        MAX_NORMAL * 1.0001, -MAX_NORMAL * 1.0001,        # just over max -> overflow
        1000.0, -5000.0, 1e6,
        np.inf, -np.inf,                                  # already-overflowed
    ], dtype=np.float64)

    # Log-uniform random magnitudes across [1e-12, 1e5], random sign.
    log_mag = rng.uniform(-12.0, 5.0, size=4000)
    mags = 10.0 ** log_mag
    signs = rng.choice([-1.0, 1.0], size=4000)
    random_vals = signs * mags

    x = np.concatenate([hand_picked, random_vals]).astype(np.float32)
    rng.shuffle(x)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "fp8_x.npy", x)


if __name__ == "__main__":
    main()
