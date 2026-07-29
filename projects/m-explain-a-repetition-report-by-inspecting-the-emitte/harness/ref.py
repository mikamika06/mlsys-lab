import numpy as np


def build_histogram(tokens):
    histogram = {}
    for token in tokens:
        histogram[token] = histogram.get(token, 0) + 1
    return histogram


def repetition_report(tokens, window, threshold):
    tokens = list(tokens)
    tail = tokens[-window:] if window > 0 else []
    tail_hist = build_histogram(tail)
    full_hist = build_histogram(tokens)

    triggered = False
    token = None
    window_count = 0
    if tail_hist:
        best = max(tail_hist.values())
        if best >= threshold:
            token = min(t for t, c in tail_hist.items() if c == best)
            window_count = best
            triggered = True

    positions = sorted(i for i, t in enumerate(tokens) if t == token) if triggered else []

    return {
        "triggered": triggered,
        "token": token,
        "window_count": window_count,
        "positions": positions,
        "histogram": full_hist,
        "total_tokens": len(tokens),
        "unique_tokens": len(full_hist),
    }


def num_predict_budget(num_predict, prompt_tokens, context_size, hard_cap):
    remaining = max(context_size - prompt_tokens, 0)
    if num_predict >= 0:
        return min(num_predict, remaining)
    if num_predict == -2:
        return remaining
    if num_predict == -1:
        return max(hard_cap - prompt_tokens, 0)
    raise ValueError(f"unsupported num_predict {num_predict}")


_rng = np.random.default_rng(42)
_RANDOM_A = _rng.integers(0, 5, size=24).tolist()
_RANDOM_B = _rng.integers(0, 9, size=40).tolist()

CASES = [
    ([0, 1, 2, 3, 4, 5, 5, 5, 5], 4, 3),
    ([1, 2, 3, 4, 5, 6, 7, 8], 4, 2),
    ([9, 9, 9, 9, 1, 2, 3, 4], 4, 3),
    ([7, 7, 3, 3], 4, 2),
    ([2, 2, 2], 3, 3),
    ([2, 2, 2], 3, 4),
    ([4, 4, 4], 10, 3),
    ([], 5, 1),
    (_RANDOM_A, 6, 4),
    (_RANDOM_B, 8, 5),
]

PREDICT_CASES = [
    (10, 5, 100, 100),
    (200, 5, 100, 100),
    (0, 5, 100, 100),
    (-2, 5, 100, 100),
    (-2, 100, 100, 100),
    (-2, 150, 100, 100),
    (-1, 5, 4096, 1_000_000),
    (-1, 5, 4096, 4096),
    (-1, 5000, 4096, 4096),
    (-2, 0, 1, 1),
]
