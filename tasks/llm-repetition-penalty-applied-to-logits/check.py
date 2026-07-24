import numpy as np

def _reference(logits, seen_tokens, penalty):
    logits = np.asarray(logits, dtype=np.float64)
    out = logits.copy()
    if len(seen_tokens) == 0:
        return out
    mask = np.zeros_like(out, dtype=bool)
    mask[list(seen_tokens)] = True
    pos_mask = mask & (out > 0)
    neg_mask = mask & (out <= 0)
    out[pos_mask] /= penalty
    out[neg_mask] *= penalty
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(10):
        vocab_size = rng.integers(5, 20)
        logits = rng.standard_normal(vocab_size).astype(np.float64)
        num_seen = rng.integers(0, vocab_size + 1)
        seen_tokens = rng.choice(vocab_size, size=num_seen, replace=False).tolist()
        penalty = rng.uniform(1.01, 3.0)

        try:
            cand = sol.apply_repetition_penalty(logits, seen_tokens, penalty)
        except Exception:
            return {"max_abs_err": float("inf")}

        if not isinstance(cand, np.ndarray) or cand.shape != logits.shape:
            return {"max_abs_err": float("inf")}
        cand = np.asarray(cand, dtype=np.float64)

        ref = _reference(logits, seen_tokens, penalty)
        err = np.max(np.abs(cand - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
