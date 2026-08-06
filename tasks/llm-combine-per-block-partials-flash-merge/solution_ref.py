import math

def merge_attention_blocks(
    m_blocks: list[list[list[float]]],
    l_blocks: list[list[list[float]]],
    O_blocks: list[list[list[float]]]
) -> list[list[float]]:
    B = len(m_blocks)
    N = len(m_blocks[0])
    d_v = len(O_blocks[0][0])
    O_final = [[0.0] * d_v for _ in range(N)]
    for i in range(N):
        m_max = m_blocks[0][i][0]
        for b in range(1, B):
            val = m_blocks[b][i][0]
            if val > m_max:
                m_max = val
        l_sum = 0.0
        weights = [0.0] * B
        for b in range(B):
            w = l_blocks[b][i][0] * math.exp(m_blocks[b][i][0] - m_max)
            weights[b] = w
            l_sum += w
        for d in range(d_v):
            o_sum = 0.0
            for b in range(B):
                o_sum += O_blocks[b][i][d] * weights[b]
            O_final[i][d] = o_sum / l_sum
    return O_final
