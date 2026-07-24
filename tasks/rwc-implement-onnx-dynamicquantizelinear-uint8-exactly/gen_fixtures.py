"""Deterministic fp32 activation-tensor fixture for the ONNX
DynamicQuantizeLinear task.

Includes tensors that are all-positive, all-negative, mixed-sign, with an
outlier, and broad random coverage -- all with genuine spread (max != min)
so y_scale never degenerates to zero.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    tensors = []
    tensors.append(np.array([-1.0, 0.0, 1.0, 2.0]))
    tensors.append(np.array([0.0, 1.0, 2.0, 3.0]))          # all non-negative
    tensors.append(np.array([-3.0, -2.0, -1.0, 0.0]))       # all non-positive
    tensors.append(np.array([-0.001, 0.0002, 0.0005]))      # tiny magnitudes
    tensors.append(np.array([-100.0, 5.0, 5.0, 5.0, 200.0]))  # outlier-dominated
    tensors.append(rng.normal(0.0, 1.0, size=64))
    tensors.append(rng.normal(0.0, 1.0, size=(8, 8)))
    tensors.append(rng.uniform(-5.0, 50.0, size=100))
    tensors.append(rng.uniform(0.0, 10.0, size=100))         # all-positive random
    tensors.append(rng.normal(0.0, 0.01, size=50))           # small-scale random

    max_len = max(t.size for t in tensors)
    N = len(tensors)
    x = np.zeros((N, max_len), dtype=np.float32)
    lengths = np.zeros((N,), dtype=np.int64)
    for i, t in enumerate(tensors):
        flat = t.astype(np.float32).ravel()
        lengths[i] = flat.size
        x[i, :flat.size] = flat

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "lengths.npy", lengths)


if __name__ == "__main__":
    main()
