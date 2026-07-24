"""Generates fixtures/gguf_w.npy: a (n_blocks, 32) weight matrix, one
ggml-style quantization block per row, with varying per-row magnitude so
the comparison isn't dominated by a single scale."""
import numpy as np

N_BLOCKS, BLOCK = 16, 32

rng = np.random.default_rng(0)
W = rng.normal(size=(N_BLOCKS, BLOCK)) * rng.uniform(0.5, 3.0, size=(N_BLOCKS, 1))

if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "gguf_w.npy", W)
    print("wrote", out / "gguf_w.npy", W.shape)
