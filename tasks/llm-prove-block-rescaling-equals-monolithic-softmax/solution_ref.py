import math

def block_rescale_softmax(logits: list[float], block_size: int) -> list[float]:
    """
    Compute softmax of `logits` using a block‑wise rescaling strategy.
    The result is identical to the monolithic softmax.
    """
    n = len(logits)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if n == 0:
        M = float('-inf')
    else:
        M = float(logits[0])
        for i in range(1, n):
            val = float(logits[i])
            if val > M:
                M = val

    exp_scaled = [0.0] * n
    denom = 0.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        m_b = float(logits[start])
        for i in range(start + 1, end):
            val = float(logits[i])
            if val > m_b:
                m_b = val

        scale_factor = math.exp(m_b - M)

        block_sum = 0.0
        for i in range(start, end):
            val = math.exp(float(logits[i]) - m_b) * scale_factor
            exp_scaled[i] = val
            block_sum += val

        denom += block_sum

    for i in range(n):
        exp_scaled[i] /= denom

    return exp_scaled
