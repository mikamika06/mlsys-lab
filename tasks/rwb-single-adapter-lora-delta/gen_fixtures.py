"""Deterministic fixture for a single LoRA adapter applied to a frozen
base layer's output.

Run from the task directory: python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(11)

    n, d, r = 10, 32, 4  # tokens, model dim, LoRA rank

    x = rng.normal(size=(n, d))
    W0 = rng.normal(scale=0.1, size=(d, d))  # frozen base weight, not saved
    base = x @ W0

    A = rng.normal(scale=0.05, size=(d, r))
    B = rng.normal(scale=0.05, size=(r, d))

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "base.npy", base)
    np.save(out / "a.npy", A)
    np.save(out / "b.npy", B)


if __name__ == "__main__":
    main()
