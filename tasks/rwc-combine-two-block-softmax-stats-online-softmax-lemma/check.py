import numpy as np

def grade(sol, fx) -> dict:
    def _stats(logits):
        m = np.max(logits)
        exp_shifted = np.exp(logits - m)
        s = np.sum(exp_shifted)
        w = np.sum(logits * exp_shifted)
        return (float(m), float(s), float(w))

    def _ref(block_a, block_b):
        m1,s1,w1 = block_a
        m2,s2,w2 = block_b
        m = max(m1,m2)
        shift1 = m1 - m
        shift2 = m2 - m
        s = s1 * np.exp(shift1) + s2 * np.exp(shift2)
        w = w1 * np.exp(shift1) + w2 * np.exp(shift2)
        return (float(m), float(s), float(w))

    max_err = 0.0
    rng = np.random.default_rng(12345)
    for _ in range(5):
        n1 = rng.integers(10, 100)
        n2 = rng.integers(10, 100)
        logits1 = rng.standard_normal(n1) * rng.uniform(1, 1000)
        logits2 = rng.standard_normal(n2) * rng.uniform(1, 1000)

        block_a = _stats(logits1)
        block_b = _stats(logits2)

        try:
            got = sol.combine_softmax_stats(block_a, block_b)
            got = tuple(float(x) for x in got)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _ref(block_a, block_b)
        err = max(abs(got[i] - ref[i]) for i in range(3))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
