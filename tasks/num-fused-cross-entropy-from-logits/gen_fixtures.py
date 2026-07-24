"""Deterministic logits/targets fixture for num-fused-cross-entropy-from-logits.

Includes rows with very large logit magnitude (up to 1e5) so that a naive,
non-max-subtracted exp() overflows to inf/nan and fails the gate.

    python3 tasks/num-fused-cross-entropy-from-logits/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"


def build():
    rng = np.random.default_rng(0)
    n, c = 200, 12
    logits = rng.uniform(-5.0, 5.0, size=(n, c)).astype(np.float64)
    logits[:10] = rng.uniform(1e4, 1e5, size=(10, c))
    logits[10:20] = rng.uniform(-1e5, -1e4, size=(10, c))
    logits[20:30] = rng.uniform(-1.0, 1.0, size=(10, c)) + rng.uniform(
        700.0, 750.0, size=(10, 1)
    )
    targets = rng.integers(0, c, size=n).astype(np.int64)
    return logits, targets


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    logits, targets = build()
    np.save(OUT / "logits.npy", logits)
    np.save(OUT / "targets.npy", targets)
    print("wrote", logits.shape, targets.shape)
