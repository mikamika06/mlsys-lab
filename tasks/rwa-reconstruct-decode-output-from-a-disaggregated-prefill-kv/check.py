import numpy as np


def _oracle_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    logits = Q @ K.T / np.sqrt(K.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (
            rng.normal(size=(3, 4)),
            rng.normal(size=(5, 4)),
            rng.normal(size=(5, 3)),
        ),
        (
            rng.normal(size=(1, 8)),
            rng.normal(size=(16, 8)),
            rng.normal(size=(16, 6)),
        ),
        (
            rng.normal(size=(7, 3)),
            rng.normal(size=(2, 3)),
            rng.normal(size=(2, 2)),
        ),
    ]

    worst = 0.0
    for Q_arr, K_arr, V_arr in cases:
        try:
            Q_list = Q_arr.tolist()
            K_list = K_arr.tolist()
            V_list = V_arr.tolist()
            payload = sol.serialize_kv(K_list, V_list)
            got = sol.decode_from_kv(Q_list, payload)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle_attention(Q_arr, K_arr, V_arr)
        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
