import numpy as np
from qknorm.config import AttentionConfig


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps)


def compute_qknorm_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: AttentionConfig,
) -> np.ndarray:
    q_norm = rms_norm(q, eps=config.eps)
    k_norm = rms_norm(k, eps=config.eps)
    scale = config.get_scale()
    scores = np.matmul(q_norm, k_norm.swapaxes(-1, -2)) * scale
    if config.softcap is not None and config.softcap > 0:
        scores = config.softcap * np.tanh(scores / config.softcap)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, v)
