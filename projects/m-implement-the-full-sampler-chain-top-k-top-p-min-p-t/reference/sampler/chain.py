import numpy as np

def apply_repetition_penalty(logits: np.ndarray, history: list[int], penalty: float, repeat_last_n: int = -1) -> np.ndarray:
    out = logits.copy()
    if repeat_last_n > 0:
        history = history[-repeat_last_n:]
    elif repeat_last_n == 0:
        history = []

    for token in set(history):
        if out[token] < 0:
            out[token] *= penalty
        else:
            out[token] /= penalty
    return out

def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    out = logits.copy()
    if k <= 0 or k >= len(out):
        return out

    threshold = np.sort(out)[-k]
    out[out < threshold] = -np.inf
    return out

def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    out = logits.copy()
    if p >= 1.0:
        return out

    sorted_indices = np.argsort(out)[::-1]
    sorted_logits = out[sorted_indices]

    probs = np.exp(sorted_logits - np.max(sorted_logits))
    probs /= np.sum(probs)

    cumsum = np.cumsum(probs)

    exceed_indices = np.where(cumsum > p)[0]
    if len(exceed_indices) > 0:
        first_exceed = exceed_indices[0]
        indices_to_remove = sorted_indices[first_exceed + 1:]
        out[indices_to_remove] = -np.inf

    return out

def apply_min_p(logits: np.ndarray, p: float) -> np.ndarray:
    out = logits.copy()
    if p <= 0.0:
        return out

    probs = np.exp(out - np.max(out))
    probs /= np.sum(probs)

    max_prob = np.max(probs)
    threshold = max_prob * p

    out[probs < threshold] = -np.inf
    return out

def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    out = logits.copy()
    if t > 0:
        out = out / t
    return out

def full_chain(logits: np.ndarray, history: list[int], penalty: float, repeat_last_n: int, k: int, top_p: float, min_p: float, t: float) -> np.ndarray:
    out = apply_repetition_penalty(logits, history, penalty, repeat_last_n)
    out = apply_top_k(out, k)
    out = apply_top_p(out, top_p)
    out = apply_min_p(out, min_p)
    out = apply_temperature(out, t)
    return out

def compare_survival(logits: np.ndarray, top_p: float, min_p: float) -> tuple[set[int], set[int]]:
    out_p = apply_top_p(logits, top_p)
    out_min = apply_min_p(logits, min_p)

    survive_p = set(np.where(~np.isinf(out_p))[0].tolist())
    survive_min = set(np.where(~np.isinf(out_min))[0].tolist())

    return survive_p, survive_min
