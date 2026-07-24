import numpy as np


def _bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    bits = bits & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _oracle(Q, K, V, dO):
    Q = _bf16(Q)
    K = _bf16(K)
    V = _bf16(V)
    dO = _bf16(dO)
    n, d = Q.shape
    scale = np.float32(1.0 / np.sqrt(d))

    S = _bf16(_bf16(Q @ K.T) * scale)
    S = _bf16(S - np.max(S, axis=1, keepdims=True))
    P = _bf16(np.exp(S))
    P = _bf16(P / np.sum(P, axis=1, keepdims=True))

    dV = _bf16(P.T @ dO)
    dP = _bf16(dO @ V.T)
    dot = _bf16(np.sum(_bf16(dP * P), axis=1, keepdims=True))
    dS = _bf16(P * _bf16(dP - dot))

    dQ = np.zeros_like(Q, dtype=np.float32)
    tile = 2
    for start in range(0, n, tile):
        end = min(n, start + tile)
        part = _bf16(dS[:, start:end] @ K[start:end])
        dQ = _bf16(dQ + _bf16(part * scale))

    dK = _bf16(_bf16(dS.T @ Q) * scale)
    return dQ.astype(np.float32), dK.astype(np.float32), dV.astype(np.float32)


def grade(sol, fx) -> dict:
    cases = [
        3,
        4,
        6,
    ]
    worst = 0.0
    rng = np.random.default_rng(1234)
    for n in cases:
        d = 4
        Q = rng.normal(size=(n, d)).astype(np.float32)
        K = rng.normal(size=(n, d)).astype(np.float32)
        V = rng.normal(size=(n, d)).astype(np.float32)
        dO = rng.normal(size=(n, d)).astype(np.float32)
        ref = _oracle(Q, K, V, dO)
        try:
            got = sol.flash_bwd_bf16(Q, K, V, dO)
            err = max(float(np.max(np.abs(a - b))) for a, b in zip(got, ref))
        except Exception:
            return {"max_abs_err": 1.0}
        worst = max(worst, err)
    return {"max_abs_err": worst}
