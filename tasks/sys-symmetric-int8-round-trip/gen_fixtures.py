"""Deterministic fp32 weight fixture for the symmetric int8 round-trip task."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # A plausible weight tensor: mostly small values clustered near zero,
    # plus a handful of much larger outliers, so the max-based scale is
    # dominated by a small fraction of the entries.
    W = rng.normal(0.0, 0.02, size=(96, 64)).astype(np.float32)
    W.ravel()[:12] = (rng.normal(0.0, 3.0, size=12)).astype(np.float32)
    W.ravel()[12] = 0.0  # exact zero must round-trip exactly
    np.save(OUT / "weights.npy", W)


if __name__ == "__main__":
    main()
