import numpy as np


def simulate_hopper_fp8(q, k, v):
    q_max = np.max(np.abs(q)) + 1e-5
    k_max = np.max(np.abs(k)) + 1e-5
    v_max = np.max(np.abs(v)) + 1e-5
    q_q = np.clip(np.round(q / q_max * 448.0), -448.0, 448.0) * (q_max / 448.0)
    k_q = np.clip(np.round(k / k_max * 448.0), -448.0, 448.0) * (k_max / 448.0)
    v_q = np.clip(np.round(v / v_max * 448.0), -448.0, 448.0) * (v_max / 448.0)
    scale = 1.0 / np.sqrt(q_q.shape[-1])
    scores = np.matmul(q_q, k_q.transpose(0, 1, 3, 2)) * scale
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp
    return np.matmul(attn, v_q).astype(np.float32)
