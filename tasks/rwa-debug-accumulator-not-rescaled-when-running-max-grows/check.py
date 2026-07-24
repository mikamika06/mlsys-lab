import numpy as np


def _dense_reference(q, K, V):
    d = K.shape[1]
    scores = (q @ K.T) / np.sqrt(d)
    scores = scores - np.max(scores)
    p = np.exp(scores)
    p = p / np.sum(p)
    return p @ V


def _cases():
    cases = []

    # Deterministic case: a strong outlier key sits in a LATER block, well
    # after the first block has already contributed non-trivial mass to O.
    # This forces at least one large accumulator rescale.
    d = 4
    K = np.zeros((28, d), dtype=np.float64)
    V = np.zeros((28, d), dtype=np.float64)
    rng0 = np.random.default_rng(1)
    K[:, :] = rng0.standard_normal((28, d)) * 0.5
    V[:, :] = rng0.standard_normal((28, d))
    K[20] = np.array([6.0, 0.0, 0.0, 0.0])  # outlier in block index 2 (block_size=8)
    V[20] = np.array([100.0, -100.0, 50.0, -50.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    cases.append((q, K, V, 8))

    # Seeded random cases with several blocks each, so the running max
    # grows more than once across the sequence.
    rng = np.random.default_rng(7)
    for _ in range(6):
        n = int(rng.integers(24, 48))
        d = int(rng.integers(3, 8))
        block_size = int(rng.integers(4, 9))
        q = rng.standard_normal(d)
        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))
        cases.append((q, K, V, block_size))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, K, V, block_size in _cases():
        ref = _dense_reference(q, K, V)
        try:
            got = np.asarray(sol.tiled_online_softmax_attention(q, K, V, block_size), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
