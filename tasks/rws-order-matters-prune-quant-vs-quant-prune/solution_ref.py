import numpy as np


def compare_prune_quant_order(W: np.ndarray, X: np.ndarray, group_size: int, sparsity: float, bits: int = 4):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)

    col_norm = np.linalg.norm(X, axis=0)
    norm_g = col_norm.reshape(ng, group_size)
    score = np.abs(Wg) * norm_g[None, :, :]

    qmax = 2 ** (bits - 1) - 1
    k_keep = max(1, int(round((1.0 - sparsity) * group_size)))

    order = np.argsort(-score, axis=2, kind="stable")
    keep_idx = order[:, :, :k_keep]
    keep_mask = np.zeros_like(score, dtype=bool)
    ri, gi = np.meshgrid(np.arange(rows), np.arange(ng), indexing="ij")
    ri_full = np.repeat(ri[:, :, None], k_keep, axis=2)
    gi_full = np.repeat(gi[:, :, None], k_keep, axis=2)
    keep_mask[ri_full, gi_full, keep_idx] = True

    # A: prune then quant -- scale derived only from the surviving weights.
    pruned = np.where(keep_mask, Wg, 0.0)
    amax_a = np.max(np.abs(pruned), axis=2)
    scale_a = amax_a / qmax
    scale_a_safe = np.where(scale_a == 0, 1.0, scale_a)
    q_a = np.clip(np.round(pruned / scale_a_safe[:, :, None]), -qmax, qmax)
    deq_a = np.where(keep_mask, q_a * scale_a_safe[:, :, None], 0.0)

    # B: quant then prune -- scale derived from the full, unpruned group.
    amax_b = np.max(np.abs(Wg), axis=2)
    scale_b = amax_b / qmax
    scale_b_safe = np.where(scale_b == 0, 1.0, scale_b)
    q_b = np.clip(np.round(Wg / scale_b_safe[:, :, None]), -qmax, qmax)
    deq_full = q_b * scale_b_safe[:, :, None]
    deq_b = np.where(keep_mask, deq_full, 0.0)

    mse_prune_then_quant = float(np.mean((Wg - deq_a) ** 2))
    mse_quant_then_prune = float(np.mean((Wg - deq_b) ** 2))
    return mse_prune_then_quant, mse_quant_then_prune
