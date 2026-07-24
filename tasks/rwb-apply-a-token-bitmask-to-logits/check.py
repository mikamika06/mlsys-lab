import numpy as np

def grade(sol, fx) -> dict:
    def _expected(logits, allowed_sets):
        mask = np.zeros_like(logits, dtype=bool)
        for i, allowed in enumerate(allowed_sets):
            mask[i, list(allowed)] = True
        masked = np.where(mask, logits, -np.inf)
        return np.argmax(masked, axis=1)

    ok = 1.0
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        n_steps = rng.integers(3, 8)
        vocab_size = rng.integers(10, 20)
        logits = rng.standard_normal((n_steps, vocab_size))
        allowed_sets = []
        for i in range(n_steps):
            k = rng.integers(1, vocab_size + 1)
            allowed = set(rng.choice(vocab_size, size=k, replace=False))
            allowed_sets.append(allowed)
        try:
            got = sol.masked_greedy(logits, allowed_sets)
            exp = _expected(logits, allowed_sets)
        except Exception:
            return {"exact_match": 0.0}
        if not np.array_equal(got, exp):
            ok = 0.0
            break
    return {"exact_match": ok}
