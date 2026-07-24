"""Generates fixtures/wanda_w.npy, wanda_x.npy, wanda_m.npy.

W: (d_out, d_in) weight matrix.
X: (d_in, n) calibration activations (column-major: d_in input features,
   n samples).
M: (d_out, d_in) binary Wanda pruning mask -- computed with the real
   Wanda metric |W_ij| * ||X_row_j||_2, 50% sparsity, pruned per OUTPUT
   ROW (Wanda's per-output comparison group).
"""
import numpy as np

D_OUT, D_IN, N = 6, 10, 20
SPARSITY = 0.5

rng = np.random.default_rng(0)
W = rng.normal(size=(D_OUT, D_IN)) * rng.uniform(0.3, 2.0, size=(1, D_IN))
X = rng.normal(size=(D_IN, N))

col_norm = np.linalg.norm(X, axis=1)  # (D_IN,), per input feature
metric = np.abs(W) * col_norm[None, :]  # (D_OUT, D_IN)

M = np.ones_like(W)
n_prune = int(round(SPARSITY * D_IN))
for i in range(D_OUT):
    order = np.argsort(metric[i], kind="stable")
    M[i, order[:n_prune]] = 0.0

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "wanda_w.npy", W)
    np.save(out / "wanda_x.npy", X)
    np.save(out / "wanda_m.npy", M)
    print("wrote wanda_w", W.shape, "wanda_x", X.shape, "wanda_m", M.shape)
