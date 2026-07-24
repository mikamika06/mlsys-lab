"""SGMV fixture: a batch of rows each tagged with a different LoRA
adapter id, and a small bank of same-rank adapters.

Run from the task directory:  python3 gen_fixtures.py
"""
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(89)
    N, d_in, d_out, r, num_adapters = 20, 12, 8, 3, 4

    x = rng.standard_normal((N, d_in))
    adapter_id = rng.integers(0, num_adapters, size=N).astype(np.int64)
    A_bank = rng.standard_normal((num_adapters, d_in, r)) * 0.5
    B_bank = rng.standard_normal((num_adapters, r, d_out)) * 0.5
    scale = rng.uniform(0.5, 2.0, size=num_adapters)

    out = pathlib.Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "adapter_id.npy", adapter_id)
    np.save(out / "A_bank.npy", A_bank)
    np.save(out / "B_bank.npy", B_bank)
    np.save(out / "scale.npy", scale)


if __name__ == "__main__":
    main()
