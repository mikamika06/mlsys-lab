import numpy as np


def _forward(Q, K, V, causal):
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    P = np.exp(scores)
    P = P / np.sum(P, axis=1, keepdims=True)
    return P @ V, np.log(np.sum(np.exp(scores), axis=1)) + np.max(Q @ K.T / np.sqrt(d) if not causal else np.where(np.triu(np.ones((n, n), dtype=bool), 1), -np.inf, Q @ K.T / np.sqrt(d)), axis=1)


def _loss(Q, K, V, dO, causal):
    O, _ = _forward(Q, K, V, causal)
    return float(np.sum(O * dO))


def _fd_grads(Q, K, V, dO, causal):
    eps = 1e-6
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for arr, grad in [(Q, dQ), (K, dK), (V, dV)]:
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps
            a = _loss(Q, K, V, dO, causal)
            arr[idx] = old - eps
            b = _loss(Q, K, V, dO, causal)
            arr[idx] = old
            grad[idx] = (a - b) / (2 * eps)
            it.iternext()
    return dQ, dK, dV


def _flash_lse(Q, K, causal):
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), 1), -np.inf, scores)
    m = np.max(scores, axis=1)
    return m + np.log(np.sum(np.exp(scores - m[:, None]), axis=1))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    total_ref = []
    total_got = []
    for n, d, m, causal in [(3, 4, 2, False), (4, 3, 3, True), (5, 2, 4, False)]:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, m))
        dO = rng.normal(size=(n, m))
        L = _flash_lse(Q, K, causal)
        try:
            got = sol.flash_backward_dq_dk_dv(Q, K, V, dO, L, causal)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _fd_grads(Q.copy(), K.copy(), V.copy(), dO, causal)
        for a, b in zip(got, ref):
            total_got.append(np.asarray(a, dtype=np.float64).ravel())
            total_ref.append(np.asarray(b, dtype=np.float64).ravel())
    a = np.concatenate(total_got)
    b = np.concatenate(total_ref)
    err = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
    return {"rel_err": float(err)}
