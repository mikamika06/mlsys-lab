import numpy as np

from mlsys import scorers


def _dot_acc(a, b, dtype):
    acc = dtype(0.0)
    for x, y in zip(a, b):
        acc = dtype(acc + dtype(dtype(x) * dtype(y)))
    return acc


def _attention_precision(Q, K, V, dtype):
    Q = np.asarray(Q, dtype=np.float64).astype(dtype)
    K = np.asarray(K, dtype=np.float64).astype(dtype)
    V = np.asarray(V, dtype=np.float64).astype(dtype)

    n, d = Q.shape
    m, dv = V.shape
    scale = dtype(1.0 / np.sqrt(d))

    O = np.zeros((n, dv), dtype=np.float64)
    for i in range(n):
        S = np.empty(m, dtype=dtype)
        for j in range(m):
            s = _dot_acc(Q[i], K[j], dtype)
            S[j] = dtype(s * scale)

        m_i = dtype(np.max(S))
        exp_vals = np.empty(m, dtype=dtype)
        for j in range(m):
            shifted = dtype(S[j] - m_i)
            exp_vals[j] = dtype(np.exp(np.float64(shifted)))

        l_i = dtype(0.0)
        for j in range(m):
            l_i = dtype(l_i + exp_vals[j])

        P = np.empty(m, dtype=dtype)
        for j in range(m):
            P[j] = dtype(exp_vals[j] / l_i)

        for k in range(dv):
            col = V[:, k]
            o = dtype(0.0)
            for j in range(m):
                o = dtype(o + dtype(P[j] * col[j]))
            O[i, k] = float(o)

    return O


def _oracle(Q, K, V):
    O_ref = _attention_precision(Q, K, V, np.float64)
    O_16 = _attention_precision(Q, K, V, np.float16)
    O_32 = _attention_precision(Q, K, V, np.float32)
    return scorers.rel_err(O_ref, O_16), scorers.rel_err(O_ref, O_32)


def grade(sol, fx) -> dict:
    Q, K, V = fx["q"], fx["k"], fx["v"]

    ref_fp16_err, ref_fp32_err = _oracle(Q, K, V)

    try:
        got = sol.fp16_vs_fp32_attention_error(Q.copy(), K.copy(), V.copy())
        got_fp16_err, got_fp32_err = float(got[0]), float(got[1])
    except Exception:
        return {"rel_err": float("inf")}

    if got_fp32_err >= got_fp16_err:
        return {"rel_err": float("inf")}
    if got_fp32_err >= 1e-3:
        return {"rel_err": float("inf")}

    discrepancy = max(
        abs(got_fp16_err - ref_fp16_err),
        abs(got_fp32_err - ref_fp32_err),
    )
    return {"rel_err": discrepancy}
