import numpy as np


def compute_divergence(base_weights, lora_a, lora_b, scale):
    w = base_weights.astype(np.float32)
    delta = (np.matmul(lora_b, lora_a) * scale).astype(np.float32)
    w_merged = w + delta
    w_merged_q = np.round(w_merged * 7.0) / 7.0
    w_q = np.round(w * 7.0) / 7.0
    w_q_adapted = w_q + delta
    diff = np.abs(w_merged_q - w_q_adapted)
    rel_err = float(np.mean(diff / (np.abs(w_merged_q) + 1e-5)))
    max_err = float(np.max(diff))
    return {"rel_err": rel_err, "max_err": max_err}
