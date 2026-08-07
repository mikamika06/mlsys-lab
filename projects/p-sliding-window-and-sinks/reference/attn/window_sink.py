import numpy as np
from attn.cache import WindowSinkKVCache


def compute_window_sink_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_sinks: int,
    window_size: int,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    L, d = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(d)

    for i in range(L):
        allowed = list(range(min(num_sinks, i + 1)))
        win_start = max(num_sinks, i - window_size + 1)
        for j in range(win_start, i + 1):
            if j not in allowed:
                allowed.append(j)
        allowed.sort()

        sub_k = k[allowed]
        sub_v = v[allowed]
        qi = q[i : i + 1]

        scores = np.dot(qi, sub_k.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out[i] = np.dot(weights, sub_v)[0]

    return out


class StreamingAttentionRunner:

    def __init__(self, num_sinks: int, window_size: int, head_dim: int):
        self.cache = WindowSinkKVCache(num_sinks, window_size, head_dim)
        self.head_dim = head_dim

    def step(self, q_tok: np.ndarray, k_tok: np.ndarray, v_tok: np.ndarray) -> np.ndarray:
        q_tok = np.asarray(q_tok, dtype=np.float64)
        k_tok = np.asarray(k_tok, dtype=np.float64)
        v_tok = np.asarray(v_tok, dtype=np.float64)

        if q_tok.ndim == 1:
            q_tok = q_tok.reshape(1, -1)
        if k_tok.ndim == 1:
            k_tok = k_tok.reshape(1, -1)
        if v_tok.ndim == 1:
            v_tok = v_tok.reshape(1, -1)

        self.cache.append(k_tok, v_tok)
        keys = self.cache.get_keys()
        vals = self.cache.get_values()

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.dot(q_tok, keys.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        out = np.dot(weights, vals)
        return out
