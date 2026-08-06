import numpy as np

def merge_weights(w_base, lora_a, lora_b, scale):
    """
    Merge adapter into base weights.
    """
    w = w_base.astype(np.float32)
    a = lora_a.astype(np.float32)
    b = lora_b.astype(np.float32)
    return w + (b @ a) * scale

def quantization_error(w_base, lora_a, lora_b, scale):
    """
    Return the relative error of 8-bit symmetric per-tensor quantization on the merged weights.
    q_scale = max(abs(w_merged)) / 127.0
    w_q = round(w_merged / q_scale) * q_scale
    rel_err = norm(w_merged - w_q) / norm(w_merged)
    """
    w_merged = merge_weights(w_base, lora_a, lora_b, scale)
    abs_max = float(np.max(np.abs(w_merged)))
    if abs_max == 0.0:
        return 0.0
    q_scale = abs_max / 127.0
    w_q = np.round(w_merged / q_scale) * q_scale
    diff = float(np.linalg.norm(w_merged - w_q))
    base = float(np.linalg.norm(w_merged))
    return diff / base if base > 0.0 else 0.0

def forward_equivalence(x, w_base, lora_a, lora_b, scale):
    """
    Return the relative error between y_adapter and y_merged.
    y_adapter = x @ w_base.T + (x @ lora_a.T @ lora_b.T) * scale
    y_merged = x @ w_merged.T
    rel_err = norm(y_adapter - y_merged) / norm(y_adapter)
    """
    x = x.astype(np.float32)
    w_base = w_base.astype(np.float32)
    lora_a = lora_a.astype(np.float32)
    lora_b = lora_b.astype(np.float32)

    y_adapter = (x @ w_base.T) + (x @ lora_a.T @ lora_b.T) * scale
    w_merged = merge_weights(w_base, lora_a, lora_b, scale)
    y_merged = x @ w_merged.T

    diff = float(np.linalg.norm(y_adapter - y_merged))
    base_norm = float(np.linalg.norm(y_adapter))
    return diff / base_norm if base_norm > 0.0 else 0.0
