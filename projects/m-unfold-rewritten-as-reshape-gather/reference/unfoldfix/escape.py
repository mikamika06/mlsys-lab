def choose_escape_hatch(error_msg):
    mapping = {
        "UNSUPPORTED_OP_UNFOLD": "rewrite_reshape_gather",
        "DYNAMIC_SHAPE_MISMATCH": "static_shape_pad",
        "QUANTIZATION_SCALE_OVERFLOW": "recompute_scale",
        "CUSTOM_KERNEL_NOT_FOUND": "fallback_aten_op",
        "ATTENTION_MASK_RANK_MISMATCH": "broadcast_mask",
        "RMSNORM_AXIS_OUT_OF_BOUNDS": "normalize_axis",
        "KV_CACHE_STRIDE_INVALID": "contiguous_cache",
        "SILU_FUSION_UNSUPPORTED": "split_silu_mul",
        "EMBEDDING_TABLE_TOO_LARGE": "shard_embedding",
        "ROPE_FREQ_BASE_INVALID": "default_rope_base"
    }
    return mapping.get(error_msg, "generic_fallback")
