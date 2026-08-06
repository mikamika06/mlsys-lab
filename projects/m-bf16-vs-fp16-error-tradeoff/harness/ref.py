import numpy as np


def generate_inputs():
    np.random.seed(42)
    q = np.random.randn(2, 4, 16, 32).astype(np.float32)
    k = np.random.randn(2, 4, 32, 32).astype(np.float32)
    v = np.random.randn(2, 4, 32, 32).astype(np.float32)
    mask = np.random.choice([0, 1], size=(2, 1, 16, 32), p=[0.2, 0.8]).astype(np.float32)
    mask[:, :, :, 0] = 0
    return q, k, v, mask


def reference_precision_error(x):
    x_fp16 = x.astype(np.float16).astype(np.float32)
    diff = np.abs(x - x_fp16)
    denom = np.maximum(np.abs(x), 1e-7)
    return float(np.mean(diff / denom))


def reference_blockwise_attention(q, k, v, mask):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    scores = np.where(mask == 0, -1e4, scores)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isneginf(row_max) | np.isnan(row_max), 0.0, row_max)
    exp_scores = np.exp(scores - row_max)
    exp_scores = np.where(mask == 0, 0.0, exp_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp_safe = np.where(sum_exp == 0.0, 1.0, sum_exp)
    attn_weights = exp_scores / sum_exp_safe
    out = np.matmul(attn_weights, v)
    return out
