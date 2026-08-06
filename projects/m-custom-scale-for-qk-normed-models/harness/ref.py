import numpy as np


def generate_test_case(seed: int, softcap: float = None, custom_scale: float = None):
    rng = np.random.RandomState(seed)
    batch = 2
    heads = 4
    seq_len = 16
    head_dim = 64

    q = rng.randn(batch, heads, seq_len, head_dim).astype(np.float64)
    k = rng.randn(batch, heads, seq_len, head_dim).astype(np.float64)
    v = rng.randn(batch, heads, seq_len, head_dim).astype(np.float64)

    return {
        "q": q,
        "k": k,
        "v": v,
        "head_dim": head_dim,
        "custom_scale": custom_scale,
        "softcap": softcap,
        "eps": 1e-6,
    }


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps)


def oracle_qknorm_attention(q, k, v, head_dim, custom_scale=None, softcap=None, eps=1e-6):
    q_norm = rms_norm(q, eps)
    k_norm = rms_norm(k, eps)
    scale = custom_scale if custom_scale is not None else (1.0 / (head_dim ** 0.5))
    scores = np.matmul(q_norm, k_norm.swapaxes(-1, -2)) * scale
    if softcap is not None and softcap > 0:
        scores = softcap * np.tanh(scores / softcap)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, v)
