import numpy as np


def _oracle(scores, values, block_size):
    scores = np.asarray(scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    m_old = -np.inf
    l_old = 0.0
    a_old = 0.0
    rows = []

    for start in range(0, len(scores), block_size):
        s = scores[start:start + block_size]
        v = values[start:start + block_size]

        m_block = np.max(s)
        block_exp = np.exp(s - m_block)
        l_block = np.sum(block_exp)
        a_block = np.sum(block_exp * v)

        m_new = max(m_old, m_block)
        if np.isneginf(m_old):
            old_scale = 0.0
        else:
            old_scale = np.exp(m_old - m_new)
        block_scale = np.exp(m_block - m_new)

        l_old = l_old * old_scale + l_block * block_scale
        a_old = a_old * old_scale + a_block * block_scale
        m_old = m_new

        rows.append([m_old, l_old, a_old])

    return np.asarray(rows, dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, 2.0, 10.0, 3.0]),
            np.array([2.0, 1.0, 4.0, 5.0]),
            2,
        ),
        (
            np.array([-5.0, -4.0, -3.0, 20.0, 21.0, 0.0]),
            np.array([1.0, 2.0, 3.0, -1.0, 4.0, 2.0]),
            3,
        ),
        (
            np.linspace(-2, 8, 11),
            np.linspace(1, 11, 11),
            4,
        ),
    ]

    worst = 0.0
    for scores, values, block_size in cases:
        try:
            got = sol.online_softmax_blocks(scores, values, block_size)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(scores, values, block_size)
        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
