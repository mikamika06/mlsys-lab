import numpy as np
from mlsys import scorers


def _ref(num_heads, trained_len, extra_len):
    total = 0.0
    count = 0
    slopes = 2.0 ** (-(np.arange(num_heads, dtype=np.float64) + 1.0) / num_heads)
    for q in range(trained_len, trained_len + extra_len):
        distances = np.arange(q, -1, -1, dtype=np.float64)
        for slope in slopes:
            logits = -slope * distances
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            total += float(np.sum(probs * distances) / (q + 1.0))
            count += 1
    return total / count


def grade(sol, fx) -> dict:
    cases = [
        (1, 8, 4),
        (2, 16, 8),
        (8, 32, 16),
        (4, 64, 32),
    ]
    refs = []
    got = []
    for args in cases:
        try:
            got.append(float(sol.alibi_extrapolation_metric(*args)))
            refs.append(_ref(*args))
        except Exception:
            return {"rel_err": 1.0}
    return {"rel_err": scorers.rel_err(np.array(refs), np.array(got))}
