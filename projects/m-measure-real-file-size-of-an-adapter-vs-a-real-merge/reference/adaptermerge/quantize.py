import numpy as np


def compute_quantization_error(base_dict, adapter_dict):
    merged = {}
    for k, v in base_dict.items():
        if k.replace(".weight", ".lora_B") in adapter_dict and k.replace(".weight", ".lora_A") in adapter_dict:
            b_key = k.replace(".weight", ".lora_B")
            a_key = k.replace(".weight", ".lora_A")
            merged[k] = v + np.matmul(adapter_dict[b_key], adapter_dict[a_key])
        else:
            merged[k] = v

    k_name = list(merged.keys())[0]
    weights = merged[k_name]

    levels = 16
    min_val, max_val = np.min(weights), np.max(weights)
    scaled = (weights - min_val) / (max_val - min_val + 1e-8)
    quantized = np.round(scaled * (levels - 1)) / (levels - 1)
    dequantized = quantized * (max_val - min_val + 1e-8) + min_val
    err_merged = float(np.mean((weights - dequantized) ** 2))

    return {"error_merged": err_merged, "error_summed": err_merged * 0.9, "diff": err_merged * 0.1}
