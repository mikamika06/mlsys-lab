import math


def online_softmax_blocks(
    scores: list[float],
    values: list[float],
    block_size: int
) -> list[list[float]]:
    m_old = -float("inf")
    l_old = 0.0
    a_old = 0.0
    rows = []

    for start in range(0, len(scores), block_size):
        s = scores[start:start + block_size]
        v = values[start:start + block_size]

        m_block = -float("inf")
        for x in s:
            if x > m_block:
                m_block = x

        e = []
        for x in s:
            e.append(math.exp(x - m_block))

        l_block = 0.0
        for val in e:
            l_block += val

        a_block = 0.0
        for i in range(len(e)):
            a_block += e[i] * v[i]

        m_new = m_old if m_old > m_block else m_block
        if math.isinf(m_old) and m_old < 0:
            old_scale = 0.0
        else:
            old_scale = math.exp(m_old - m_new)
        block_scale = math.exp(m_block - m_new)

        l_old = l_old * old_scale + l_block * block_scale
        a_old = a_old * old_scale + a_block * block_scale
        m_old = m_new

        rows.append([m_old, l_old, a_old])

    return rows
