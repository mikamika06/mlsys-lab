import math

def scaled_dot_product_attention(
    Q: list[list[list[float]]],
    K: list[list[list[float]]],
    V: list[list[list[float]]],
    *,
    causal: bool = False
) -> list[list[list[float]]]:
    """
    Compute scaled dot‑product attention with optional causal masking.

    Parameters
    ----------
    Q, K : list[list[list[float]]]
        Query and key tensors of shape (B, N, d_k).
    V : list[list[list[float]]]
        Value tensor of shape (B, N, d_v).
    causal : bool, default False
        If True, apply a lower‑triangular causal mask.

    Returns
    -------
    out : list[list[list[float]]]
        Attention output of shape (B, N, d_v).
    """
    B = len(Q)
    N = len(Q[0])
    d_k = len(Q[0][0])
    d_v = len(V[0][0])

    scores = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(B)]
    scale = math.sqrt(d_k)

    for b in range(B):
        for i in range(N):
            for j in range(N):
                acc = 0.0
                for k in range(d_k):
                    acc += Q[b][i][k] * K[b][j][k]
                scores[b][i][j] = acc / scale

    if causal:
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if j > i:
                        scores[b][i][j] = -float("inf")

    attn_weights = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(B)]
    for b in range(B):
        for i in range(N):
            max_val = scores[b][i][0]
            for j in range(1, N):
                if scores[b][i][j] > max_val:
                    max_val = scores[b][i][j]

            sum_exp = 0.0
            for j in range(N):
                val = math.exp(scores[b][i][j] - max_val)
                attn_weights[b][i][j] = val
                sum_exp += val

            for j in range(N):
                attn_weights[b][i][j] /= sum_exp

    out = [[[0.0 for _ in range(d_v)] for _ in range(N)] for _ in range(B)]
    for b in range(B):
        for i in range(N):
            for j in range(d_v):
                acc = 0.0
                for k in range(N):
                    acc += attn_weights[b][i][k] * V[b][k][j]
                out[b][i][j] = acc

    return out
