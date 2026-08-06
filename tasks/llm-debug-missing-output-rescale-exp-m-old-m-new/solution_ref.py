import math


def online_softmax_update(
    m_old,
    l_old,
    O_old,
    m_block,
    l_block,
    O_block,
):
    m_new = max(m_old, m_block)
    alpha = math.exp(m_old - m_new)
    beta = math.exp(m_block - m_new)

    l_new = alpha * l_old + beta * l_block
    O_new = [
        (alpha * l_old * o_old + beta * l_block * o_block) / l_new
        for o_old, o_block in zip(O_old, O_block)
    ]

    return float(m_new), float(l_new), [float(x) for x in O_new]
