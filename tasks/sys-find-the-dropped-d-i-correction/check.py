import numpy as np


def _forward(Q, K, V, scale):
    S = (Q @ K.T) * scale
    m = np.max(S, axis=1, keepdims=True)
    ex = np.exp(S - m)
    l = np.sum(ex, axis=1, keepdims=True)
    P = ex / l
    O = P @ V
    L = (m + np.log(l)).reshape(-1)
    return O, L


def _numeric_grad(Q, K, V, dO, scale, h=1e-4):
    def loss(Qc, Kc, Vc):
        O, _ = _forward(Qc, Kc, Vc, scale)
        return float(np.sum(O * dO))

    def grad_wrt(arr, build_loss):
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + h
            lp = build_loss()
            arr[idx] = orig - h
            lm = build_loss()
            arr[idx] = orig
            g[idx] = (lp - lm) / (2 * h)
        return g

    gQ = grad_wrt(Q, lambda: loss(Q, K, V))
    gK = grad_wrt(K, lambda: loss(Q, K, V))
    gV = grad_wrt(V, lambda: loss(Q, K, V))
    return gQ, gK, gV


def grade(sol, fx) -> dict:
    """
    For several seeded random (Q, K, V, dO) instances: runs the real
    (numpy-only) softmax-attention forward pass to obtain O and the row
    logsumexp L, computes the ground-truth dQ/dK/dV via CENTRAL FINITE
    DIFFERENCES on the scalar loss sum(O(Q,K,V) * dO) (an oracle that never
    calls the submission), and compares them element-wise to the
    submission's flash_attention_backward output. Reports the worst-case
    max abs error across all trials and tensors.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        n = int(rng.integers(4, 7))
        d = int(rng.integers(2, 5))
        scale = 1.0 / np.sqrt(d)
        Q = rng.standard_normal((n, d)) * 0.5
        K = rng.standard_normal((n, d)) * 0.5
        V = rng.standard_normal((n, d)) * 0.5
        dO = rng.standard_normal((n, d)) * 0.5

        O, L = _forward(Q, K, V, scale)
        gQ, gK, gV = _numeric_grad(Q, K, V, dO, scale)

        try:
            dQ, dK, dV = sol.flash_attention_backward(
                Q.copy(), K.copy(), V.copy(), O.copy(), L.copy(), dO.copy(), scale
            )
            dQ = np.asarray(dQ, dtype=np.float64)
            dK = np.asarray(dK, dtype=np.float64)
            dV = np.asarray(dV, dtype=np.float64)
        except Exception:
            return {"max_abs_err": 1e9}

        if dQ.shape != Q.shape or dK.shape != K.shape or dV.shape != V.shape:
            return {"max_abs_err": 1e9}

        err = max(
            float(np.max(np.abs(dQ - gQ))),
            float(np.max(np.abs(dK - gK))),
            float(np.max(np.abs(dV - gV))),
        )
        worst = max(worst, err)
    return {"max_abs_err": worst}
