import numpy as np

def grade(sol, fx) -> dict:
    ok = 1.0
    rng = np.random.default_rng(42)
    for _ in range(10):
        batch = rng.integers(2, 20)
        vocab = rng.integers(50, 200)
        logits = rng.standard_normal((batch, vocab))
        try:
            got = sol.greedy_argmax_next_token(logits)
            ref = np.argmax(logits, axis=1)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, np.ndarray) or got.shape != (batch,):
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
