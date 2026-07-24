import numpy as np

def _reference(logits, token_counts, freq_penalty, presence_penalty):
    logits = np.asarray(logits, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.int64)
    presence_mask = (token_counts > 0).astype(np.float64)
    penalty = token_counts * freq_penalty + presence_mask * presence_penalty
    return logits - penalty

def grade(sol, fx) -> dict:
    # generate deterministic test cases
    rng = np.random.default_rng(42)
    ok = True
    max_err = 0.0
    for _ in range(5):
        vocab_size = rng.integers(10, 50)
        logits = rng.standard_normal(vocab_size).astype(np.float64)
        token_counts = rng.integers(0, 6, size=vocab_size).astype(np.int64)
        freq_penalty = rng.uniform(0.0, 0.5)
        presence_penalty = rng.uniform(0.0, 0.5)
        try:
            got = sol.apply_frequency_presence_penalty(
                logits,
                token_counts,
                freq_penalty,
                presence_penalty
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference(logits, token_counts, freq_penalty, presence_penalty)
        err = np.max(np.abs(got - ref))
        if err > max_err:
            max_err = err
        if err > 1e-6:
            ok = False
    return {"max_abs_err": float(max_err) if ok else float("inf")}
