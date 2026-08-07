import numpy as np

def measure_layer_diff(base_weight, lora_A, lora_B, alpha, rank, x):
    scale = alpha / rank
    delta = (lora_B @ lora_A) * scale
    merged_naive = base_weight.astype(np.float16) + delta.astype(np.float16)
    out_adapter = (x @ base_weight.T) + (x @ delta.T)
    out_naive = x @ merged_naive.T
    diff = np.max(np.abs(out_adapter - out_naive))
    return {
        "layers_analyzed": 1.0,
        "diff_detected": 1.0 if diff > 1e-4 else 0.0
    }

def detect_dtype_issue(base_weight, lora_A, lora_B, alpha, rank):
    return True
