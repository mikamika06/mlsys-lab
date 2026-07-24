"""Four deterministic 'layers' of deliberately differing weight-magnitude
scale, for the global-vs-per-tensor pruning sparsity-drift task.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(51)

    shapes = [(40, 40), (30, 50), (64, 20), (48, 48)]
    scales = [0.01, 2.0, 0.3, 1.0]  # wildly different per-layer magnitude
    layers = [rng.standard_normal(s) * sc for s, sc in zip(shapes, scales)]

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    for i, layer in enumerate(layers):
        np.save(out / f"layer{i}.npy", layer)


if __name__ == "__main__":
    main()
