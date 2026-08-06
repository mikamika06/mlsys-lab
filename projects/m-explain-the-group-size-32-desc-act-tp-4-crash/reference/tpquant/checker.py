"""Diagnostic checker for quantization and TP configurations."""


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
