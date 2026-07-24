"""Four layer weight tensors at deliberately different magnitude scales, so
that a single GLOBAL magnitude threshold prunes them very unevenly (a
per-layer threshold would instead prune the same fraction from each layer).

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)

    layer0 = (rng.standard_normal((10, 10)) * 0.001).astype(np.float64)   # tiny scale
    layer1 = (rng.standard_normal((20, 5)) * 1.0).astype(np.float64)      # unit scale
    layer2 = (rng.standard_normal((8, 8)) * 50.0).astype(np.float64)      # large scale
    layer3 = (rng.standard_normal((15, 3)) * 0.05).astype(np.float64)     # small scale

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "layer0.npy", layer0)
    np.save(out / "layer1.npy", layer1)
    np.save(out / "layer2.npy", layer2)
    np.save(out / "layer3.npy", layer3)


if __name__ == "__main__":
    main()
