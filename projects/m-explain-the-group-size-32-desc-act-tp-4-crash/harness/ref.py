"""Reference oracle and data generator for TPQuant harness."""
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


def prepare_tp_linear(in_features, out_features, group_size, tp_size, perm, scales, mode):
    analysis = analyze_tp_slice(in_features, group_size, tp_size, perm)
    total_groups = in_features // group_size
    g_per_rank = total_groups // tp_size
    k_per_rank = in_features // tp_size

    if mode == "validate_only":
        if not analysis["is_safe"]:
            raise ValueError("Incompatible quantization layout for Tensor Parallelism")
        g_idx = build_g_idx(in_features, group_size, perm)
        ranks = []
        for r in range(tp_size):
            slice_g = g_idx[r * k_per_rank : (r + 1) * k_per_rank]
            local_g = slice_g - r * g_per_rank
            slice_s = scales[r * g_per_rank : (r + 1) * g_per_rank]
            ranks.append({"rank": r, "g_idx": local_g, "scales": slice_s})
        return {"ranks": ranks, "mode": mode}

    if mode == "replicate_scales":
        g_idx = build_g_idx(in_features, group_size, perm)
        ranks = []
        for r in range(tp_size):
            slice_g = g_idx[r * k_per_rank : (r + 1) * k_per_rank]
            ranks.append({"rank": r, "g_idx": slice_g, "scales": scales.copy()})
        return {"ranks": ranks, "mode": mode}

    if mode == "disable_desc_act":
        g_idx = build_g_idx(in_features, group_size, None)
        ranks = []
        for r in range(tp_size):
            slice_g = g_idx[r * k_per_rank : (r + 1) * k_per_rank]
            local_g = slice_g - r * g_per_rank
            slice_s = scales[r * g_per_rank : (r + 1) * g_per_rank]
            ranks.append({"rank": r, "g_idx": local_g, "scales": slice_s})
        return {"ranks": ranks, "mode": mode}

    raise ValueError(f"Unknown mode: {mode}")


def diagnose_config(in_features, group_size, tp_size, desc_act):
    if tp_size > 1 and desc_act:
        return {
            "cause": "permuted_g_idx_cross_rank_scale_oob",
            "has_oob": True,
            "has_fragmentation": True,
            "recommended_mode": "replicate_scales",
        }
    return {
        "cause": "none",
        "has_oob": False,
        "has_fragmentation": False,
        "recommended_mode": "validate_only",
    }


def make_configs():
    rng = np.random.RandomState(42)
    configs = []
    p0 = rng.permutation(128)
    configs.append({
        "in_features": 128, "out_features": 64, "group_size": 32, "tp_size": 4,
        "perm": p0, "desc_act": True
    })
    configs.append({
        "in_features": 128, "out_features": 64, "group_size": 32, "tp_size": 4,
        "perm": None, "desc_act": False
    })
    configs.append({
        "in_features": 128, "out_features": 64, "group_size": 32, "tp_size": 1,
        "perm": rng.permutation(128), "desc_act": True
    })
    configs.append({
        "in_features": 256, "out_features": 128, "group_size": 64, "tp_size": 2,
        "perm": rng.permutation(256), "desc_act": True
    })
    return configs

CONFIGS = make_configs()
