import math
import numpy as np


def compare_prune_quant_order(W: np.ndarray, X: np.ndarray, group_size: int, sparsity: float, bits: int = 4):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)

    rows_X, cols_X = X.shape
    col_norm = np.zeros(cols_X, dtype=np.float64)
    for c in range(cols_X):
        sum_sq = 0.0
        for r in range(rows_X):
            val = X[r, c]
            sum_sq += val * val
        col_norm[c] = math.sqrt(sum_sq)

    norm_g = col_norm.reshape(ng, group_size)
    score = np.zeros((rows, ng, group_size), dtype=np.float64)
    for r in range(rows):
        for g in range(ng):
            for i in range(group_size):
                w_val = Wg[r, g, i]
                abs_w = w_val if w_val >= 0.0 else -w_val
                score[r, g, i] = abs_w * norm_g[g, i]

    qmax = 2 ** (bits - 1) - 1
    k_keep = max(1, int(round((1.0 - sparsity) * group_size)))

    order = np.zeros((rows, ng, group_size), dtype=int)
    for r in range(rows):
        for g in range(ng):
            s_vals = score[r, g, :]
            sorted_pairs = sorted(enumerate(s_vals), key=lambda item: (-item[1], item[0]))
            for rank, (orig_idx, val) in enumerate(sorted_pairs):
                order[r, g, rank] = orig_idx

    keep_mask = np.zeros((rows, ng, group_size), dtype=bool)
    for r in range(rows):
        for g in range(ng):
            for k in range(k_keep):
                idx = order[r, g, k]
                keep_mask[r, g, idx] = True

    pruned = np.zeros((rows, ng, group_size), dtype=np.float64)
    amax_a = np.zeros((rows, ng), dtype=np.float64)
    scale_a = np.zeros((rows, ng), dtype=np.float64)
    scale_a_safe = np.zeros((rows, ng), dtype=np.float64)
    q_a = np.zeros((rows, ng, group_size), dtype=np.float64)
    deq_a = np.zeros((rows, ng, group_size), dtype=np.float64)

    for r in range(rows):
        for g in range(ng):
            max_val = 0.0
            for i in range(group_size):
                val = Wg[r, g, i] if keep_mask[r, g, i] else 0.0
                pruned[r, g, i] = val
                abs_val = val if val >= 0.0 else -val
                if abs_val > max_val:
                    max_val = abs_val
            amax_a[r, g] = max_val
            s_a = max_val / qmax
            scale_a[r, g] = s_a
            s_a_safe = 1.0 if s_a == 0.0 else s_a
            scale_a_safe[r, g] = s_a_safe

            for i in range(group_size):
                p_val = pruned[r, g, i]
                quot = p_val / s_a_safe
                rnd = round(quot)
                clipped = max(-qmax, min(qmax, rnd))
                q_a[r, g, i] = clipped
                if keep_mask[r, g, i]:
                    deq_a[r, g, i] = clipped * s_a_safe
                else:
                    deq_a[r, g, i] = 0.0

    amax_b = np.zeros((rows, ng), dtype=np.float64)
    scale_b = np.zeros((rows, ng), dtype=np.float64)
    scale_b_safe = np.zeros((rows, ng), dtype=np.float64)
    q_b = np.zeros((rows, ng, group_size), dtype=np.float64)
    deq_b = np.zeros((rows, ng, group_size), dtype=np.float64)

    for r in range(rows):
        for g in range(ng):
            max_val = 0.0
            for i in range(group_size):
                val = Wg[r, g, i]
                abs_val = val if val >= 0.0 else -val
                if abs_val > max_val:
                    max_val = abs_val
            amax_b[r, g] = max_val
            s_b = max_val / qmax
            scale_b[r, g] = s_b
            s_b_safe = 1.0 if s_b == 0.0 else s_b
            scale_b_safe[r, g] = s_b_safe

            for i in range(group_size):
                w_val = Wg[r, g, i]
                quot = w_val / s_b_safe
                rnd = round(quot)
                clipped = max(-qmax, min(qmax, rnd))
                q_b[r, g, i] = clipped
                deq_full_val = clipped * s_b_safe
                if keep_mask[r, g, i]:
                    deq_b[r, g, i] = deq_full_val
                else:
                    deq_b[r, g, i] = 0.0

    sum_sq_a = 0.0
    sum_sq_b = 0.0
    count = 0
    for r in range(rows):
        for g in range(ng):
            for i in range(group_size):
                diff_a = Wg[r, g, i] - deq_a[r, g, i]
                sum_sq_a += diff_a * diff_a
                diff_b = Wg[r, g, i] - deq_b[r, g, i]
                sum_sq_b += diff_b * diff_b
                count += 1

    mse_prune_then_quant = float(sum_sq_a / count)
    mse_quant_then_prune = float(sum_sq_b / count)
    return mse_prune_then_quant, mse_quant_then_prune
