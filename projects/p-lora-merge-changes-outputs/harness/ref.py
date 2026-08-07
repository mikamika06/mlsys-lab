import numpy as np

def generate_test_data(seed=42):
    np.random.seed(seed)
    base_weight = np.random.randn(64, 64).astype(np.float32) * 0.1
    lora_A = np.random.randn(8, 64).astype(np.float32) * 0.5
    lora_B = np.random.randn(64, 8).astype(np.float32) * 0.5
    alpha = 16.0
    rank = 8
    x = np.random.randn(10, 64).astype(np.float32)
    return base_weight, lora_A, lora_B, alpha, rank, x

def oracle_layer_diff(base_weight, lora_A, lora_B, alpha, rank, x):
    scale = alpha / rank
    delta = (lora_B @ lora_A) * scale
    merged_naive = base_weight.astype(np.float16) + delta.astype(np.float16)
    merged_safe = base_weight + delta.astype(np.float32)
    out_adapter = (x @ base_weight.T) + (x @ delta.T)
    out_naive = x @ merged_naive.T
    out_safe = x @ merged_safe.T
    err_naive = np.max(np.abs(out_adapter - out_naive))
    err_safe = np.max(np.abs(out_adapter - out_safe))
    return {
        "layers_analyzed": 1.0,
        "diff_detected": 1.0 if err_naive > 1e-4 else 0.0,
        "dtype_bug_found": 1.0,
        "scale_correct": 1.0,
        "merge_safe": 1.0 if err_safe < 1e-5 else 0.0,
        "max_error_low": 1.0 if err_safe < 1e-5 else 0.0
    }
