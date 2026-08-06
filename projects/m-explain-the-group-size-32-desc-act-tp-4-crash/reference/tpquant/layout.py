"""Layout analysis for tensor parallel quantized layers."""
import numpy as np


def build_g_idx(in_features, group_size, perm=None):
    if perm is None:
        return np.arange(in_features, dtype=np.int64) // group_size
    return np.asanyarray(perm, dtype=np.int64) // group_size


def analyze_tp_slice(in_features, group_size, tp_size, perm=None):
    g_idx = build_g_idx(in_features, group_size, perm)
    total_groups = in_features // group_size
    g_per_rank = total_groups // tp_size
    k_per_rank = in_features // tp_size

    oob_ranks = []
    for r in range(tp_size):
        slice_g = g_idx[r * k_per_rank : (r + 1) * k_per_rank]
        exp_min = r * g_per_rank
        exp_max = (r + 1) * g_per_rank - 1
        if np.any(slice_g < exp_min) or np.any(slice_g > exp_max):
            oob_ranks.append(r)

    rank_of_elem = np.arange(in_features) // k_per_rank
    frag_count = 0
    for g in range(total_groups):
        ranks_for_g = set(rank_of_elem[g_idx == g])
        if len(ranks_for_g) > 1:
            frag_count += 1

    is_safe = (len(oob_ranks) == 0) and (frag_count == 0)
    return {
        "in_features": in_features,
        "group_size": group_size,
        "tp_size": tp_size,
        "k_per_rank": k_per_rank,
        "total_groups": total_groups,
        "oob_ranks": oob_ranks,
        "fragmented_groups_count": frag_count,
        "is_safe": is_safe,
    }
