import math


def stream_softmax_row_chunks(
    logits: list[list[float]],
    chunk_size: int
) -> list[list[float]]:
    rows = len(logits)
    cols = len(logits[0]) if rows > 0 else 0
    out = [[0.0] * cols for _ in range(rows)]

    for r in range(rows):
        m = -float('inf')
        l = 0.0
        row = logits[r]

        for start in range(0, cols, chunk_size):
            end = min(start + chunk_size, cols)
            mc = -float('inf')
            for c in range(start, end):
                val = row[c]
                if val > mc:
                    mc = val

            m_new = m if m > mc else mc

            term1 = l * math.exp(m - m_new) if m != -float('inf') else 0.0
            term2 = 0.0
            for c in range(start, end):
                term2 += math.exp(row[c] - m_new)

            l = term1 + term2
            m = m_new

        for start in range(0, cols, chunk_size):
            end = min(start + chunk_size, cols)
            for c in range(start, end):
                out[r][c] = math.exp(row[c] - m) / l

    return out
