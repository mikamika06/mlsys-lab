import numpy as np
from attn.mask import generate_sliding_window_mask


def windowed_attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    window_size: int,
    num_sinks: int = 0,
    is_causal: bool = True,
) -> np.ndarray:
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    head_dim = query.shape[-1]

    mask = generate_sliding_window_mask(
        q_len=q_len,
        kv_len=kv_len,
        window_size=window_size,
        num_sinks=num_sinks,
        is_causal=is_causal,
    )

    scores = np.matmul(query, key.swapaxes(-1, -2)) / np.sqrt(head_dim)
    scores = np.where(mask, scores, -1e9)

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    exp_scores = np.where(mask, exp_scores, 0.0)

    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(weights, value)
