import numpy as np


def compute_offset(n_q: int, n_kv: int, alignment: str) -> int:
    if alignment == "bottom_right":
        return n_kv - n_q
    elif alignment == "top_left":
        return 0
    else:
        raise ValueError(f"Unknown alignment: {alignment}")


def compute_causal_mask(n_q: int, n_kv: int, alignment: str = "bottom_right") -> np.ndarray:
    offset = compute_offset(n_q, n_kv, alignment)
    q_idx = np.arange(n_q)[:, None]
    kv_idx = np.arange(n_kv)[None, :]
    return kv_idx <= (q_idx + offset)


def sdpa_rectangular_causal(q: np.ndarray, k: np.ndarray, v: np.ndarray, alignment: str = "bottom_right") -> np.ndarray:
    n_q = q.shape[-2]
    n_kv = k.shape[-2]
    d_k = q.shape[-1]

    scale = 1.0 / np.sqrt(d_k)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale

    mask = compute_causal_mask(n_q, n_kv, alignment=alignment)
    scores = np.where(mask, scores, -1e9)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    exp_scores = np.where(mask, exp_scores, 0.0)

    weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
    return np.matmul(weights, v)


def flash_attn_sim(q: np.ndarray, k: np.ndarray, v: np.ndarray, is_causal: bool = True, alignment: str = "bottom_right") -> np.ndarray:
    if not is_causal:
        d_k = q.shape[-1]
        scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(d_k)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        return np.matmul(weights, v)

    return sdpa_rectangular_causal(q, k, v, alignment=alignment)


TEST_SHAPES = [
    (16, 64),
    (32, 128),
    (64, 64),
    (1, 256),
]
