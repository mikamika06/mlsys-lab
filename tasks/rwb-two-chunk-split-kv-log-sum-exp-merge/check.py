import numpy as np


def _full_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scale = 1.0 / np.sqrt(q.shape[0])
    s = (k @ q) * scale
    m = np.max(s)
    w = np.exp(s - m)
    w = w / np.sum(w)
    return w @ v


def _chunk_partial(q, k_chunk, v_chunk):
    scale = 1.0 / np.sqrt(q.shape[0])
    s = (k_chunk @ q) * scale
    m_i = np.max(s)
    exp_s = np.exp(s - m_i)
    l_i = np.sum(exp_s)
    o_i = (exp_s @ v_chunk) / l_i
    L_i = m_i + np.log(l_i)
    return L_i, o_i


def _oracle(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    N = k.shape[0]
    split = N // 2

    L1, O1 = _chunk_partial(q, k[:split], v[:split])
    L2, O2 = _chunk_partial(q, k[split:], v[split:])

    m = max(L1, L2)
    w1 = np.exp(L1 - m)
    w2 = np.exp(L2 - m)
    return (O1 * w1 + O2 * w2) / (w1 + w2)


def grade(sol, fx) -> dict:
    q, k, v = fx["q"], fx["k"], fx["v"]

    ref = _oracle(q, k, v)
    ref_full = _full_attention(q, k, v)
    assert np.max(np.abs(ref - ref_full)) < 1e-9  # internal sanity check

    try:
        got = sol.two_chunk_split_kv_merge(q.copy(), k.copy(), v.copy())
        got = np.asarray(got, dtype=np.float64)
    except Exception:
        return {"max_abs_err": float("inf")}

    if got.shape != ref.shape:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": float(np.max(np.abs(got - ref)))}
