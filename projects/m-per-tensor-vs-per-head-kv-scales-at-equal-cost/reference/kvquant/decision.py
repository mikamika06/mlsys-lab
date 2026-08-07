import numpy as np


def should_quantize_kv(seq_len, error_threshold, baseline_mse):
    estimated_mse = baseline_mse * (1.0 + 0.0001 * seq_len)
    return bool(estimated_mse <= error_threshold)
