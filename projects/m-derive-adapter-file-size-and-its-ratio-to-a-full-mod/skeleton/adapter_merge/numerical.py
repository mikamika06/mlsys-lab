import numpy as np

def merge_weights(w_base, lora_a, lora_b, scale):
    """
    Merge adapter into base weights.
    """
    raise NotImplementedError

def quantization_error(w_base, lora_a, lora_b, scale):
    """
    Return the relative error of 8-bit symmetric per-tensor quantization on the merged weights.
    q_scale = max(abs(w_merged)) / 127.0
    w_q = round(w_merged / q_scale) * q_scale
    rel_err = norm(w_merged - w_q) / norm(w_merged)
    """
    raise NotImplementedError

def forward_equivalence(x, w_base, lora_a, lora_b, scale):
    """
    Return the relative error between y_adapter and y_merged.
    y_adapter = x @ w_base.T + (x @ lora_a.T @ lora_b.T) * scale
    y_merged = x @ w_merged.T
    rel_err = norm(y_adapter - y_merged) / norm(y_adapter)
    """
    raise NotImplementedError
