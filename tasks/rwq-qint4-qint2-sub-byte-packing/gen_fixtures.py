"""Generates fixtures/qnt_w.npy: a (d_out, d_in) weight matrix with
d_in divisible by 4, so both 2-codes-per-byte (qint4) and
4-codes-per-byte (qint2) packing apply cleanly."""
import numpy as np

D_OUT, D_IN = 6, 32

rng = np.random.default_rng(0)
W = rng.normal(size=(D_OUT, D_IN)) * rng.uniform(0.5, 3.0, size=(D_OUT, 1))

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "qnt_w.npy", W)
    print("wrote", out / "qnt_w.npy", W.shape)
