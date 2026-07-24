import numpy as np


def _ref(m_old, l_old, O_old, m_block, l_block, O_block):
    m_new = max(m_old, m_block)
    alpha = np.exp(m_old - m_new)
    beta = np.exp(m_block - m_new)
    l_new = alpha * l_old + beta * l_block
    O_new = (
        alpha * l_old * np.asarray(O_old, dtype=np.float64)
        + beta * l_block * np.asarray(O_block, dtype=np.float64)
    ) / l_new
    return float(m_new), float(l_new), O_new


def grade(sol, fx) -> dict:
    cases = [
        (
            1.0,
            2.0,
            np.array([3.0, 4.0]),
            3.0,
            1.5,
            np.array([5.0, 6.0]),
        ),
        (
            -2.5,
            4.0,
            np.array([0.2, -1.0, 3.0]),
            -1.0,
            2.0,
            np.array([1.5, 2.5, -0.5]),
        ),
        (
            5.0,
            1.0,
            np.array([10.0, 20.0]),
            5.0,
            3.0,
            np.array([-2.0, 8.0]),
        ),
        (
            8.0,
            0.7,
            np.array([4.0, -3.0, 2.0, 1.0]),
            2.0,
            5.0,
            np.array([9.0, 1.0, -4.0, 2.0]),
        ),
    ]

    worst = 0.0
    for case in cases:
        ref_m, ref_l, ref_o = _ref(*case)
        try:
            got_m, got_l, got_o = sol.online_softmax_update(*case)
            got_o = np.asarray(got_o, dtype=np.float64)
            err = max(
                abs(float(got_m) - ref_m),
                abs(float(got_l) - ref_l),
                float(np.max(np.abs(got_o - ref_o))),
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
