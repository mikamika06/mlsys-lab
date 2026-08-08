import numpy as np
from palettize.kmeans import k_means_quantize
from palettize.layout import compute_tensor_bytes


def compare_scalar_vector(weights: np.ndarray, n_bits: int, vector_dim: int) -> dict:
    flat = weights.astype(np.float32).ravel()
    cb_scalar, idx_scalar = k_means_quantize(flat, n_bits)
    recon_scalar = cb_scalar[idx_scalar]
    scalar_mse = float(np.mean((flat - recon_scalar) ** 2))
    scalar_bytes = compute_tensor_bytes(flat.size, n_bits, len(cb_scalar), vector_dim=1)

    n_vectors = flat.size // vector_dim
    vec_data = flat[: n_vectors * vector_dim].reshape(n_vectors, vector_dim)

    norms = np.linalg.norm(vec_data, axis=1)
    cb_vec_norms, idx_vec = k_means_quantize(norms, n_bits)

    safe_norms = np.where(norms == 0, 1.0, norms)
    directions = vec_data / safe_norms[:, None]
    recon_vector = (directions * cb_vec_norms[idx_vec][:, None]).ravel()

    vector_mse = float(np.mean((flat[: n_vectors * vector_dim] - recon_vector) ** 2))
    vector_bytes = compute_tensor_bytes(flat.size, n_bits, len(cb_vec_norms), vector_dim=1)

    return {
        "scalar_mse": scalar_mse,
        "scalar_bytes": scalar_bytes,
        "vector_mse": vector_mse,
        "vector_bytes": vector_bytes,
    }
