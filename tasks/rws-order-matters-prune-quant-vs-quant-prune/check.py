import numpy as np

GROUP_SIZE = 16
SPARSITY = 0.3
BITS = 4


def _oracle(W: np.ndarray, X: np.ndarray, group_size: int, sparsity: float, bits: int):
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

    # A: prune then quant (scale derived from the SURVIVING weights only)
    pruned = np.where(keep_mask, Wg, 0.0)
    amax_a = np.max(np.abs(pruned), axis=2)
    scale_a = amax_a / qmax
    scale_a_safe = np.where(scale_a == 0, 1.0, scale_a)
    q_a = np.clip(np.round(pruned / scale_a_safe[:, :, None]), -qmax, qmax)
    deq_a = np.where(keep_mask, q_a * scale_a_safe[:, :, None], 0.0)

    # B: quant then prune (scale derived from the FULL original group)
    amax_b = np.max(np.abs(Wg), axis=2)
    scale_b = amax_b / qmax
    scale_b_safe = np.where(scale_b == 0, 1.0, scale_b)
    q_b = np.clip(np.round(Wg / scale_b_safe[:, :, None]), -qmax, qmax)
    deq_full = q_b * scale_b_safe[:, :, None]
    deq_b = np.where(keep_mask, deq_full, 0.0)

    mse_a = float(np.mean((Wg - deq_a) ** 2))
    mse_b = float(np.mean((Wg - deq_b) ** 2))
    return mse_a, mse_b


def _fail():
    return {
        "mse_prune_then_quant_err": float("inf"),
        "mse_quant_then_prune_err": float("inf"),
        "order_correct": 0.0,
    }


def grade(sol, fx) -> dict:
    W = fx["opw_w"]
    X = fx["opw_x"]
    mse_a_ref, mse_b_ref = _oracle(W, X, GROUP_SIZE, SPARSITY, BITS)

    try:
        out = sol.compare_prune_quant_order(W.copy(), X.copy(), GROUP_SIZE, SPARSITY, BITS)
    except Exception:
        return _fail()

    try:
        mse_a_got = float(out[0])
        mse_b_got = float(out[1])
    except Exception:
        return _fail()

    if not (np.isfinite(mse_a_got) and np.isfinite(mse_b_got)):
        return _fail()

    return {
        "mse_prune_then_quant_err": abs(mse_a_got - mse_a_ref),
        "mse_quant_then_prune_err": abs(mse_b_got - mse_b_ref),
        "order_correct": 1.0 if mse_a_got <= mse_b_got else 0.0,
    }
