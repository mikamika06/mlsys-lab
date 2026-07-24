"""Generates fixtures/gptq_w.npy and fixtures/gptq_x.npy.

W: (d_out, d_in) layer weight matrix.
X: (n_cal, d_in) calibration activations, built with low-rank structure
(correlated columns) plus noise so the calibration Hessian H = X^T X has
non-trivial off-diagonal terms -- this is what makes GPTQ's H^-1 error
compensation actually beat plain round-to-nearest (RTN) quantization.
"""
import numpy as np

d_out, d_in, n_cal, k = 6, 8, 60, 3

rng = np.random.default_rng(0)
A = rng.normal(size=(n_cal, k))
B = rng.normal(size=(k, d_in))
X = A @ B + 0.3 * rng.normal(size=(n_cal, d_in))
W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 2.0, size=(1, d_in))

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "gptq_w.npy", W.astype(np.float64))
    np.save(out / "gptq_x.npy", X.astype(np.float64))
    print("wrote", out / "gptq_w.npy", W.shape)
    print("wrote", out / "gptq_x.npy", X.shape)
