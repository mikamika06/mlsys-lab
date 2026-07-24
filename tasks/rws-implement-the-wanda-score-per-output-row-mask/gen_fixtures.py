"""Deterministic weight matrix and calibration activations for the Wanda
score + per-row mask task.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np

D_OUT, D_IN = 10, 20
N_SAMPLES = 40


def main() -> None:
    rng = np.random.default_rng(0)

    W = rng.standard_normal((D_OUT, D_IN)).astype(np.float64)
    # Give input channels noticeably different activation scales, so the
    # activation-aware score differs from a plain |W| magnitude ranking.
    channel_scale = rng.uniform(0.2, 5.0, size=D_IN)
    X = (rng.standard_normal((N_SAMPLES, D_IN)) * channel_scale[None, :]).astype(np.float64)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "ww.npy", W)
    np.save(out / "wx.npy", X)


if __name__ == "__main__":
    main()
