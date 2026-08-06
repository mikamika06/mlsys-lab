import numpy as np

def _reference(logits, seen_tokens, penalty):
    out = list(logits)
    seen_set = set(seen_tokens)
    for i in range(len(out)):
        if i in seen_set:
            val = out[i]
            if val > 0:
                out[i] = val / penalty
            elif val <= 0:
                out[i] = val * penalty
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(10):
        vocab_size = int(rng.integers(5, 20))
        logits_arr = rng.standard_normal(vocab_size).astype(float)
        logits = logits_arr.tolist()
        num_seen = int(rng.integers(0, vocab_size + 1))
        seen_tokens = rng.choice(vocab_size, size=num_seen, replace=False).tolist()
        penalty = float(rng.uniform(1.01, 3.0))

        try:
            cand = sol.apply_repetition_penalty(logits, seen_tokens, penalty)
        except Exception:
            return {"max_abs_err": float("inf")}

        if not isinstance(cand, list) or len(cand) != len(logits):
            return {"max_abs_err": float("inf")}

        ref = _reference(logits, seen_tokens, penalty)
        err = max(abs(c - r) for c, r in zip(cand, ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
