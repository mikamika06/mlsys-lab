import numpy as np
from edge_export.sparsity import apply_nm_pruning, apply_unstructured_pruning


def palettize_weights(weights: np.ndarray, num_bits: int) -> np.ndarray:
    if num_bits <= 0:
        return np.zeros_like(weights)
    num_levels = 2 ** num_bits
    nz_mask = weights != 0
    if not np.any(nz_mask):
        return weights.copy()
    nz_vals = weights[nz_mask]
    min_v, max_v = np.min(nz_vals), np.max(nz_vals)
    if min_v == max_v:
        return weights.copy()
    bins = np.linspace(min_v, max_v, num_levels)
    idx = np.clip(np.round((nz_vals - min_v) / (max_v - min_v) * (num_levels - 1)).astype(int), 0, num_levels - 1)
    quantized_nz = bins[idx]
    out = np.zeros_like(weights)
    out[nz_mask] = quantized_nz
    return out


def evaluate_joint_error(weights: np.ndarray, n: int, m: int, num_bits: int, use_nm: bool) -> float:
    if use_nm:
        pruned = apply_nm_pruning(weights, n, m)
    else:
        sparsity_ratio = (m - n) / float(m)
        pruned = apply_unstructured_pruning(weights, sparsity_ratio)
    palettized = palettize_weights(pruned, num_bits)
    return float(np.mean((weights - palettized) ** 2))


def find_optimal_joint_budget(weights: np.ndarray, max_effective_bits: float, bit_options: list) -> dict:
    best = None
    best_mse = float("inf")
    n_options = [(2, 4), (1, 4), (2, 8), (4, 8)]
    for n, m in n_options:
        density = n / float(m)
        for bits in bit_options:
            eff_bits = density * bits
            if eff_bits > max_effective_bits + 1e-6:
                continue
            for use_nm in [True, False]:
                mse = evaluate_joint_error(weights, n, m, bits, use_nm)
                if mse < best_mse:
                    best_mse = mse
                    best = {
                        "n": n,
                        "m": m,
                        "bits": bits,
                        "use_nm": use_nm,
                        "effective_bits": float(eff_bits),
                        "mse": mse,
                    }
    return best
