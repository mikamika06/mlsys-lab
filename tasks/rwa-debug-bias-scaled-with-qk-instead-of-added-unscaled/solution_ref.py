import math


def sdpa_with_additive_bias(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    bias: list[list[float]],
    scale: float,
) -> list[list[float]]:
    """Scaled dot-product attention with an additive bias (padding mask,
    ALiBi, relative position bias, ...).

    Matches the real formula used by e.g. torch.nn.functional's
    scaled_dot_product_attention with a float attn_mask: the QK^T product
    is scaled FIRST, and the bias is added AFTER scaling, unscaled. The
    bias represents a fixed logit offset (e.g. "-1e9 to mask this key" or
    an ALiBi slope*distance term) -- it must not be shrunk by `scale`.

    q: (n_q, d), k: (n_k, d), v: (n_k, d_v), bias: (n_q, n_k).
    Returns (n_q, d_v).
    """
    n_q = len(q)
    d = len(q[0])
    n_k = len(k)
    d_v = len(v[0])

    logits = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            dot = 0.0
            for l in range(d):
                dot += q[i][l] * k[j][l]
            val = dot * scale + bias[i][j]
            row.append(val)
        logits.append(row)

    attn = []
    for row in logits:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        sum_exp = sum(exps)
        attn.append([e / sum_exp for e in exps])

    out = []
    for i in range(n_q):
        out_row = []
        for j in range(d_v):
            val = 0.0
            for l in range(n_k):
                val += attn[i][l] * v[l][j]
            out_row.append(val)
        out.append(out_row)

    return out
