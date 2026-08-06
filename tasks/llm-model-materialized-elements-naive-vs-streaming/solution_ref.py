import math

def softmax_streaming(logits: list[list[float]]) -> list[list[float]]:
    """Stable vectorized softmax applied row‑wise."""
    rows = len(logits)
    cols = len(logits[0]) if rows > 0 else 0
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        max_val = logits[i][0]
        for j in range(1, cols):
            if logits[i][j] > max_val:
                max_val = logits[i][j]

        row_sum = 0.0
        for j in range(cols):
            val = math.exp(logits[i][j] - max_val)
            out[i][j] = val
            row_sum += val

        for j in range(cols):
            out[i][j] /= row_sum

    return out
