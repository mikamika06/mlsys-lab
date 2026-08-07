import numpy as np


def _forward(q, k, v):
    s = q @ k.T
    m = np.max(s, axis=1, keepdims=True)
    p = np.exp(s - m)
    l = np.sum(p, axis=1, keepdims=True)
    p = p / l
    out = p @ v
    return out, m, l


def _loss(q, k, v, do):
    out, _, _ = _forward(q, k, v)
    return float(np.sum(out * do))


def _finite_diff(q, k, v, do, which, eps=1e-5):
    arrs = [q.copy(), k.copy(), v.copy()]
    grad = np.zeros_like(arrs[which], dtype=np.float64)
    for idx in np.ndindex(arrs[which].shape):
        plus = [x.copy() for x in arrs]
        minus = [x.copy() for x in arrs]
        plus[which][idx] += eps
        minus[which][idx] -= eps
        grad[idx] = (_loss(plus[0], plus[1], plus[2], do) -
                     _loss(minus[0], minus[1], minus[2], do)) / (2 * eps)
    return grad


def _oracle(q, k, v, do):
    return (
        _finite_diff(q, k, v, do, 0),
        _finite_diff(q, k, v, do, 1),
        _finite_diff(q, k, v, do, 2),
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    q = rng.normal(size=(3, 2)).astype(np.float64)
    k = rng.normal(size=(3, 2)).astype(np.float64)
    v = rng.normal(size=(3, 2)).astype(np.float64)
    do = rng.normal(size=(3, 2)).astype(np.float64)

    _, m, l = _forward(q, k, v)
    ref = _oracle(q, k, v, do)

    q_list = q.tolist()
    k_list = k.tolist()
    v_list = v.tolist()
    do_list = do.tolist()
    m_list = m.tolist()
    l_list = l.tolist()

    try:
        got = sol.flash_attention_backward(q_list, k_list, v_list, do_list, m_list, l_list)
        got_0 = np.asarray(got[0], dtype=np.float64)
        got_1 = np.asarray(got[1], dtype=np.float64)
        got_2 = np.asarray(got[2], dtype=np.float64)
        err = max(
            float(np.max(np.abs(got_0 - ref[0]))),
            float(np.max(np.abs(got_1 - ref[1]))),
            float(np.max(np.abs(got_2 - ref[2]))),
        )
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
