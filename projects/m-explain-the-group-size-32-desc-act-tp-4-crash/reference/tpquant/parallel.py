"""Parallel dispatch preparation for quantized layers."""
import numpy as np
from tpquant.layout import analyze_tp_slice, build_g_idx


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
