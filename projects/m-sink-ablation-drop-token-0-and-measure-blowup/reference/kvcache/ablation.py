import numpy as np


def compute_attention_with_sink(k, v, q):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, v)


def measure_sink_ablation_blowup(k, v, q):
    baseline = compute_attention_with_sink(k, v, q)
    k_ablated = k[:, :, 1:, :]
    v_ablated = v[:, :, 1:, :]
    ablated = compute_attention_with_sink(k_ablated, v_ablated, q)
    diff = np.abs(baseline - ablated)
    denom = np.abs(baseline) + 1e-5
    rel_err = np.max(diff / denom)
    norm_ratio = np.linalg.norm(ablated) / (np.linalg.norm(baseline) + 1e-8)
    return float(rel_err), float(norm_ratio)
