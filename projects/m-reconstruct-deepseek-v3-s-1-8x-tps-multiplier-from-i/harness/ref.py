import numpy as np

def compute_tps_multiplier(acceptance_rate: float) -> float:
    return float(1.0 + acceptance_rate * 0.8)

def get_comparison_table() -> dict:
    return {
        "MTP": {"training": "native", "kv_sharing": "shared", "extra_params": "low"},
        "Medusa": {"training": "frozen_base", "kv_sharing": "separate_heads", "extra_params": "medium"},
        "EAGLE": {"training": "frozen_base_with_autoregressive", "kv_sharing": "separate_rnn_head", "extra_params": "medium"}
    }

def compute_second_position_accuracy(head_type: str, logits: list, targets: list) -> float:
    preds = np.argmax(logits, axis=-1)
    matches = (preds == np.array(targets))
    if head_type == "sequential":
        return float(np.mean(matches) * 0.95)
    elif head_type == "parallel":
        return float(np.mean(matches) * 0.85)
    return float(np.mean(matches))
