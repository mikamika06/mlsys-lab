def get_comparison_table() -> dict:
    """Return structured comparison table for MTP, Medusa, and EAGLE."""
    return {
        "MTP": {"training": "native", "kv_sharing": "shared", "extra_params": "low"},
        "Medusa": {"training": "frozen_base", "kv_sharing": "separate_heads", "extra_params": "medium"},
        "EAGLE": {"training": "frozen_base_with_autoregressive", "kv_sharing": "separate_rnn_head", "extra_params": "medium"}
    }
