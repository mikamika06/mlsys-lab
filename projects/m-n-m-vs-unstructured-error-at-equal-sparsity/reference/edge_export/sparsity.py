import numpy as np


def apply_unstructured_pruning(weights: np.ndarray, sparsity_ratio: float) -> np.ndarray:
    if sparsity_ratio <= 0.0:
        return weights.copy()
    if sparsity_ratio >= 1.0:
        return np.zeros_like(weights)
    flat = weights.flatten()
    k = int(np.round(len(flat) * sparsity_ratio))
    if k == 0:
        return weights.copy()
    threshold = np.partition(np.abs(flat), k - 1)[k - 1]
    mask = np.abs(weights) > threshold
    if np.sum(~mask) < k:
        eq_idx = np.where(np.abs(weights) == threshold)
        needed = k - np.sum(np.abs(weights) < threshold)
        mask_flat = mask.flatten()
        eq_flat = np.where(np.abs(flat) == threshold)[0]
        mask_flat[eq_flat[:needed]] = False
        mask = mask_flat.reshape(weights.shape)
    return weights * mask


def apply_nm_pruning(weights: np.ndarray, n: int, m: int) -> np.ndarray:
    shape = weights.shape
    flat = weights.reshape(-1, m)
    abs_flat = np.abs(flat)
    partition_idx = m - n
    thresholds = np.partition(abs_flat, partition_idx - 1, axis=1)[:, partition_idx - 1:partition_idx]
    mask = abs_flat >= thresholds
    counts = np.sum(mask, axis=1)
    for i in range(len(flat)):
        if counts[i] > n:
            over = counts[i] - n
            eq = np.where(abs_flat[i] == thresholds[i][0])[0]
            mask[i, eq[:over]] = False
    return (flat * mask).reshape(shape)


def compare_sparsity_error(weights: np.ndarray, n: int, m: int) -> dict:
    sparsity_ratio = (m - n) / float(m)
    unstruct_w = apply_unstructured_pruning(weights, sparsity_ratio)
    nm_w = apply_nm_pruning(weights, n, m)
    unstruct_mse = float(np.mean((weights - unstruct_w) ** 2))
    nm_mse = float(np.mean((weights - nm_w) ** 2))
    return {
        "sparsity_ratio": float(sparsity_ratio),
        "unstructured_mse": unstruct_mse,
        "nm_mse": nm_mse,
    }
