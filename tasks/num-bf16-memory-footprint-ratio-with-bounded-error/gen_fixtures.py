"""Deterministic fp32 weight fixture for the bf16 packing task."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # a plausible weight tensor: mostly small values, a few large outliers,
    # plus exact zeros and exact powers of two so ties-to-even actually fires.
    W = rng.normal(0.0, 0.05, size=(128, 64)).astype(np.float32)
    W.ravel()[:16] = np.float32(0.0)
    W.ravel()[16:32] = (2.0 ** rng.integers(-8, 8, size=16)).astype(np.float32)
    W.ravel()[32:40] = rng.normal(0.0, 4.0, size=8).astype(np.float32)
    np.save(OUT / "W.npy", W)


if __name__ == "__main__":
    main()
