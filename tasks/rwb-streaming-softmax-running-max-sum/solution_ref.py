import math


def streaming_softmax(scores: list[float], chunk_size: int) -> list[float]:
    """
    Compute softmax(scores) by streaming through it in chunks of at most
    `chunk_size` elements, maintaining a running max `m` and running sum
    `l` (rescaling `l` whenever a new chunk raises the max), then
    normalizing every element with the final (m, l).
    """
    n = len(scores)

    m = -float("inf")
    l = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_max = -float("inf")
        for i in range(start, end):
            val = scores[i]
            if val > chunk_max:
                chunk_max = val
        m_new = max(m, chunk_max)
        alpha = 0.0 if m == -float("inf") else math.exp(m - m_new)
        chunk_sum = 0.0
        for i in range(start, end):
            chunk_sum += math.exp(scores[i] - m_new)
        l = l * alpha + chunk_sum
        m = m_new

    result = [0.0] * n
    for i in range(n):
        result[i] = math.exp(scores[i] - m) / l
    return result
