import numpy as np


def _oracle(Q, K, V):
    n_q, n, d = Q.shape
    n_kv = K.shape[0]
    g = n_q // n_kv
    scale = 1.0 / np.sqrt(d)
    O = np.zeros_like(Q)
    for h in range(n_q):
        kv = h // g
        S = (Q[h] @ K[kv].T) * scale
        S = S - np.max(S, axis=1, keepdims=True)
        P = np.exp(S)
        P = P / np.sum(P, axis=1, keepdims=True)
        O[h] = P @ V[kv]
    return O


def grade(sol, fx) -> dict:
    """
    Builds several seeded random GQA instances -- including MHA (n_kv==n_q),
    MQA (n_kv==1), and general GQA (1 < n_kv < n_q) -- computes the
    reference output per query head directly with NumPy (looping over
    heads, indexing KV head h // (n_q // n_kv), numerically-stable
    softmax), and compares it element-wise to the submission's output.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    configs = [
        (4, 4),  # MHA
        (4, 1),  # MQA
        (6, 2),
        (6, 3),
        (8, 4),
        (9, 3),
    ]
    for n_q, n_kv in configs:
        n = int(rng.integers(3, 9))
        d = int(rng.integers(2, 6))
        Q = rng.standard_normal((n_q, n, d)) * 0.7
        K = rng.standard_normal((n_kv, n, d)) * 0.7
        V = rng.standard_normal((n_kv, n, d)) * 0.7

        expected = _oracle(Q, K, V)

        try:
            got = sol.gqa_attention(Q.copy(), K.copy(), V.copy())
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": 1e9}

        if got.shape != expected.shape:
            return {"max_abs_err": 1e9}

        worst = max(worst, float(np.max(np.abs(got - expected))))
    return {"max_abs_err": worst}
