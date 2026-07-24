import numpy as np


def _oracle(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[1]
    n_kv = K.shape[1]
    r = n_q // n_kv

    K_exp = np.repeat(K, r, axis=1)  # blocked broadcast: (batch, n_q, seq_k, d)
    V_exp = np.repeat(V, r, axis=1)

    d = Q.shape[-1]
    scores = (Q @ K_exp.swapaxes(-2, -1)) / np.sqrt(d)  # (batch, n_q, seq_q, seq_k)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    return weights @ V_exp  # (batch, n_q, seq_q, d_v)


def _random_cases(rng):
    # (batch, n_q, n_kv, seq_q, seq_k, d)
    specs = [
        (2, 4, 4, 5, 5, 8),    # degenerate: n_kv == n_q
        (1, 8, 2, 3, 4, 4),    # r = 4
        (3, 6, 3, 2, 2, 16),   # r = 2
        (1, 12, 1, 2, 3, 4),   # MQA extreme: r = n_q
        (2, 9, 3, 4, 2, 6),    # r = 3
    ]
    cases = []
    for batch, n_q, n_kv, seq_q, seq_k, d in specs:
        Q = rng.standard_normal((batch, n_q, seq_q, d))
        K = rng.standard_normal((batch, n_kv, seq_k, d))
        V = rng.standard_normal((batch, n_kv, seq_k, d))
        cases.append((Q, K, V))
    return cases


def _grouping_probe_case(rng):
    # Single key/value token per KV head + one-hot V rows: since softmax
    # over a single key is always exactly 1, the output for query head h
    # is exactly the one-hot row of kv_group(h) = h // r. This pins down
    # the actual grouping used, independent of QK score values.
    n_kv = 3
    r = 2
    n_q = n_kv * r
    batch = 1
    seq_q = 2
    seq_k = 1
    d = 4
    d_v = n_kv

    Q = rng.standard_normal((batch, n_q, seq_q, d))
    K = rng.standard_normal((batch, n_kv, seq_k, d))
    V = np.zeros((batch, n_kv, seq_k, d_v))
    for kv in range(n_kv):
        V[0, kv, 0, kv] = 1.0

    return Q, K, V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = _random_cases(rng)
    cases.append(_grouping_probe_case(rng))

    max_err = 0.0
    for Q, K, V in cases:
        ref = _oracle(Q, K, V)
        try:
            got = sol.enable_gqa_broadcast_attention(Q.copy(), K.copy(), V.copy())
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
