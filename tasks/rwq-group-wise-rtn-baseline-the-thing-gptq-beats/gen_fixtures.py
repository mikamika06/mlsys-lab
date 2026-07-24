"""Generates fixtures/gptq_w.npy: a (d_out, d_in) weight matrix whose
groups of GROUP_SIZE columns have deliberately different magnitude ranges
(so a per-tensor or wrong-axis absmax bug produces visibly different codes
than the correct per-row, per-group absmax)."""
import numpy as np

GROUP_SIZE = 128
d_out, n_groups = 5, 2
d_in = n_groups * GROUP_SIZE

rng = np.random.default_rng(0)
W = np.empty((d_out, d_in), dtype=np.float64)
for g in range(n_groups):
    mag = rng.uniform(0.5, 6.0)
    W[:, g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = rng.normal(
        scale=mag, size=(d_out, GROUP_SIZE)
    )

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "gptq_w.npy", W)
    print("wrote", out / "gptq_w.npy", W.shape)
