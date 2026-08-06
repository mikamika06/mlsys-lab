import numpy as np


def apply_temperature_and_topp(logits: np.ndarray, temperature: float, top_p: float) -> np.ndarray:
    if temperature <= 0.0:
        res = np.zeros_like(logits, dtype=np.float64)
        res[np.argmax(logits)] = 1.0
        return res
    scaled = logits / temperature
    max_l = np.max(scaled)
    exp_l = np.exp(scaled - max_l)
    probs = exp_l / np.sum(exp_l)
    if top_p >= 1.0:
        return probs
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cum_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cum_probs, top_p)
    keep_idx = sorted_idx[: cutoff + 1]
    truncated = np.zeros_like(probs)
    truncated[keep_idx] = probs[keep_idx]
    s = np.sum(truncated)
    if s > 0:
        truncated /= s
    else:
        truncated[sorted_idx[0]] = 1.0
    return truncated


def compute_acceptance_prob(p_logits: np.ndarray, q_logits: np.ndarray, token_id: int, temperature: float, top_p_target: float, top_p_draft: float) -> float:
    p_dist = apply_temperature_and_topp(p_logits, temperature, top_p_target)
    q_dist = apply_temperature_and_topp(q_logits, temperature, top_p_draft)
    p_val = p_dist[token_id]
    q_val = q_dist[token_id]
    if q_val <= 0.0:
        return 0.0
    return min(1.0, p_val / q_val)


def sample_residual(p_logits: np.ndarray, q_logits: np.ndarray, temperature: float, top_p_target: float, top_p_draft: float) -> int:
    p_dist = apply_temperature_and_topp(p_logits, temperature, top_p_target)
    q_dist = apply_temperature_and_topp(q_logits, temperature, top_p_draft)
    diff = np.maximum(0.0, p_dist - q_dist)
    s = np.sum(diff)
    if s > 0:
        res_dist = diff / s
    else:
        res_dist = p_dist
    return int(np.random.choice(len(p_dist), p=res_dist))
