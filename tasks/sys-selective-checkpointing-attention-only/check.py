import numpy as np


def _attention_loss(Q, K, V, G):
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    out = weights @ V
    return float(np.sum(out * G))


def _finite_diff(Q, K, V, G, target):
    h = 1e-6
    arr = target.copy()
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + h
        if target is Q:
            plus = _attention_loss(arr, K, V, G)
        elif target is K:
            plus = _attention_loss(Q, arr, V, G)
        else:
            plus = _attention_loss(Q, K, arr, G)
        arr[idx] = old - h
        if target is Q:
            minus = _attention_loss(arr, K, V, G)
        elif target is K:
            minus = _attention_loss(Q, arr, V, G)
        else:
            minus = _attention_loss(Q, K, arr, G)
        arr[idx] = old
        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


def _full_checkpoint_bytes(Q, K, V):
    n, d = Q.shape
    m = K.shape[0]
    scores = n * m * 8
    probs = n * m * 8
    output = n * d * 8
    return Q.nbytes + K.nbytes + V.nbytes + scores + probs + output


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(3, 2)).astype(np.float64)
    K = rng.normal(size=(4, 2)).astype(np.float64)
    V = rng.normal(size=(4, 2)).astype(np.float64)
    G = rng.normal(size=(3, 2)).astype(np.float64)

    ref_q = _finite_diff(Q, K, V, G, Q)
    ref_k = _finite_diff(Q, K, V, G, K)
    ref_v = _finite_diff(Q, K, V, G, V)

    try:
        got_q, got_k, got_v, memory = sol.attention_checkpoint(Q, K, V, G)
        err = max(
            float(np.max(np.abs(got_q - ref_q))),
            float(np.max(np.abs(got_k - ref_k))),
            float(np.max(np.abs(got_v - ref_v))),
        )
        full_bytes = _full_checkpoint_bytes(Q, K, V)
        reduction = float(full_bytes > int(memory))
    except Exception:
        err = float("inf")
        reduction = 0.0

    return {
        "max_abs_err": err,
        "memory_reduction": reduction,
    }
