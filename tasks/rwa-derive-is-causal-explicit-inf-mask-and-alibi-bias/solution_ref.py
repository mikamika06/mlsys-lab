import math


def apply_attention_bias(logits, is_causal=False, alibi_slope=None):
    q = len(logits)
    k = len(logits[0]) if q > 0 else 0
    out = [[float(val) for val in row] for row in logits]

    for i in range(q):
        for j in range(k):
            if alibi_slope is not None:
                out[i][j] += float(alibi_slope) * (j - i)
            if is_causal and j > i:
                out[i][j] = -math.inf

    return out
