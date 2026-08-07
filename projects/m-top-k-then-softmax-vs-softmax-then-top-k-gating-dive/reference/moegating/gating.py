import numpy as np


def top_k_then_softmax(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    topk_indices = np.argsort(-logits, axis=-1)[:, :k]
    batch_indices = np.arange(logits.shape[0])[:, None]
    topk_logits = logits[batch_indices, topk_indices]
    exp_logits = np.exp(topk_logits - np.max(topk_logits, axis=-1, keepdims=True))
    weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return weights, topk_indices


def softmax_then_top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    topk_indices = np.argsort(-probs, axis=-1)[:, :k]
    batch_indices = np.arange(probs.shape[0])[:, None]
    weights = probs[batch_indices, topk_indices]
    return weights, topk_indices


def analyze_gating_divergence(
    logits: np.ndarray, k: int
) -> dict[str, float | np.ndarray]:
    w1, idx1 = top_k_then_softmax(logits, k)
    w2, idx2 = softmax_then_top_k(logits, k)
    max_diff = float(np.max(np.abs(w1 - w2)))
    dot_product = np.sum(w1 * w2, axis=-1)
    norm_w1 = np.linalg.norm(w1, axis=-1)
    norm_w2 = np.linalg.norm(w2, axis=-1)
    cosine_sim = dot_product / (norm_w1 * norm_w2 + 1e-12)
    return {
        "max_abs_diff": max_diff,
        "cosine_sim": cosine_sim,
        "weights_topk_first": w1,
        "weights_softmax_first": w2,
        "indices_topk_first": idx1,
        "indices_softmax_first": idx2,
    }
