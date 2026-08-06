import math

def masked_softmax(logits: list[list[float]], mask: list[list[int]]) -> list[list[float]]:
    """
    Compute softmax probabilities with additive -inf masking.

    Parameters
    ----------
    logits : list[list[float]]
        Raw attention scores of shape (batch, seq_len).
    mask : list[list[int]]
        Integer mask of the same shape. Positions where mask==0 are masked out.

    Returns
    -------
    probs : list[list[float]]
        Softmax probabilities with masked entries set to zero and each row summing to one.
    """
    batch = len(logits)
    seq_len = len(logits[0])
    probs = [[0.0 for _ in range(seq_len)] for _ in range(batch)]

    for i in range(batch):
        max_val = -float('inf')
        for j in range(seq_len):
            val = float(logits[i][j]) if mask[i][j] else -float('inf')
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        exp_vals = []
        for j in range(seq_len):
            if mask[i][j]:
                e = math.exp(float(logits[i][j]) - max_val)
            else:
                e = math.exp(-float('inf') - max_val)
            exp_vals.append(e)
            sum_exp += e

        for j in range(seq_len):
            probs[i][j] = exp_vals[j] / sum_exp

    return probs
