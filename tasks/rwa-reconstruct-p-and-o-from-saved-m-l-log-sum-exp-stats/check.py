import numpy as np


def _true_S(Q, K):
    d = Q.shape[-1]
    return (Q @ K.T) / np.sqrt(d)


def _oracle(Q, K, V, m, l):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    S = _true_S(Q, K)
    P = np.exp(S - m[:, None]) / l[:, None]
    O = P @ V
    return P, O


def _true_stats(S):
    m = np.max(S, axis=1)
    l = np.sum(np.exp(S - m[:, None]), axis=1)
    return m, l


def _cases(rng):
    cases = []
    shapes = [(4, 4, 3, 2), (6, 5, 4, 3), (3, 3, 8, 5), (10, 7, 2, 4)]
    for n, k, d, dv in shapes:
        Q = rng.standard_normal((n, d))
        K = rng.standard_normal((k, d))
        V = rng.standard_normal((k, dv))
        S = _true_S(Q, K)

        m_true, l_true = _true_stats(S)
        cases.append((Q, K, V, m_true, l_true))

        # Robustness case: shift m up by a random positive offset and
        # recompute l to stay consistent with that shift. P/O are
        # mathematically unchanged (softmax is shift-invariant), so this
        # catches implementations that ignore the supplied m/l and
        # recompute their own (mismatched) stats internally.
        offset = rng.uniform(0.5, 5.0, size=n)
        m_shift = m_true + offset
        l_shift = np.sum(np.exp(S - m_shift[:, None]), axis=1)
        cases.append((Q, K, V, m_shift, l_shift))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0

    for Q, K, V, m_stat, l_stat in _cases(rng):
        ref_P, ref_O = _oracle(Q, K, V, m_stat, l_stat)

        try:
            got = sol.flash_forward_reconstruct(
                Q.copy(), K.copy(), V.copy(), m_stat.copy(), l_stat.copy()
            )
            got_P, got_O = got
            got_P = np.asarray(got_P, dtype=np.float64)
            got_O = np.asarray(got_O, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got_P.shape != ref_P.shape or got_O.shape != ref_O.shape:
            return {"max_abs_err": float("inf")}

        row_sum_err = float(np.max(np.abs(np.sum(got_P, axis=1) - 1.0)))
        p_err = float(np.max(np.abs(got_P - ref_P)))
        o_err = float(np.max(np.abs(got_O - ref_O)))

        max_err = max(max_err, row_sum_err, p_err, o_err)

    return {"max_abs_err": max_err}
