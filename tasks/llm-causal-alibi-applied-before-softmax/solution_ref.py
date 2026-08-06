import math

def causal_alibi_logits(logits: list[list[float]], alibi_bias: list[list[float]]) -> list[list[float]]:
    """
    Compute attention probabilities with a causal mask and ALiBi bias.

    Parameters
    ----------
    logits : list[list[float]]
        Raw attention logits of shape (seq_len, seq_len).
    alibi_bias : list[list[float]]
        Linear bias to add to each logit, same shape as ``logits``.

    Returns
    -------
    probs : list[list[float]]
        Row-wise softmax probabilities after applying the causal mask and bias.
    """
    rows = len(logits)
    if rows == 0:
        return []
    cols = len(logits[0])

    if any(len(row) != cols for row in logits) or any(len(row) != cols for row in alibi_bias) or len(alibi_bias) != rows:
        raise ValueError("logits and alibi_bias must have identical shapes")

    probs = [[0.0] * cols for _ in range(rows)]

    for i in range(rows):
        max_val = -float('inf')
        for j in range(cols):
            if j <= i:
                val = float(logits[i][j]) + float(alibi_bias[i][j])
            else:
                val = -float('inf')
            if val > max_val:
                max_val = val

        exp_vals = [0.0] * cols
        exp_sum = 0.0
        for j in range(cols):
            if j <= i:
                val = float(logits[i][j]) + float(alibi_bias[i][j])
                e = math.exp(val - max_val)
            else:
                e = 0.0
            exp_vals[j] = e
            exp_sum += e

        for j in range(cols):
            probs[i][j] = exp_vals[j] / exp_sum

    return probs
