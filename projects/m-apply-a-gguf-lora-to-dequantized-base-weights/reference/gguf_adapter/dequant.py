import numpy as np


def apply_lora_to_dequantized_base(dequantized_base: dict, delta_dict: dict) -> dict:
    fused_weights = {}
    for layer_name, base_weight in dequantized_base.items():
        base_arr = np.asarray(base_weight, dtype=np.float32)
        if layer_name in delta_dict:
            delta = np.asarray(delta_dict[layer_name]["delta"], dtype=np.float32)
            fused_weights[layer_name] = base_arr + delta
        else:
            fused_weights[layer_name] = base_arr.copy()
    return fused_weights
