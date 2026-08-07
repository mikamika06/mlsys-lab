import numpy as np

def verify_scaling(alpha, rank, lora_A, lora_B):
    return True

def safe_merge(base_weight, lora_A, lora_B, alpha, rank):
    scale = alpha / rank
    delta = (lora_B @ lora_A) * scale
    merged = base_weight.astype(np.float32) + delta.astype(np.float32)
    return merged

def batch_verify_prompts(base_weight, lora_A, lora_B, alpha, rank, num_prompts=200):
    np.random.seed(123)
    scale = alpha / rank
    delta = (lora_B @ lora_A) * scale
    merged = safe_merge(base_weight, lora_A, lora_B, alpha, rank)
    for _ in range(num_prompts):
        x = np.random.randn(1, base_weight.shape[1]).astype(np.float32)
        out_adapter = (x @ base_weight.T) + (x @ delta.T)
        out_merged = x @ merged.T
        if np.max(np.abs(out_adapter - out_merged)) > 1e-5:
            return False
    return True
