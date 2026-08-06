import numpy as np
from palettize.kmeans import kmeans_palettize


def compare_scalar_vector(tensor: np.ndarray, n_bits: int) -> dict[str, float]:
    c_s, l_s = kmeans_palettize(tensor, n_bits, vector_length=1)
    recon_s = c_s[l_s].ravel()[:tensor.size].reshape(tensor.shape)
    mse_s = float(np.mean((tensor.astype(np.float32) - recon_s) ** 2))

    c_v, l_v = kmeans_palettize(tensor, n_bits, vector_length=4)
    flat_len = tensor.size
    pad_len = (4 - (flat_len % 4)) % 4
    recon_v = c_v[l_v].ravel()[:flat_len + pad_len][:flat_len].reshape(tensor.shape)
    mse_v = float(np.mean((tensor.astype(np.float32) - recon_v) ** 2))

    return {"scalar_mse": mse_s, "vector_mse": mse_v}
