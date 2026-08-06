import math


def gqa_limit_nkv_1(
    Q: list[list[list[float]]],
    K: list[list[list[float]]],
    V: list[list[list[float]]],
) -> list[list[list[float]]]:
    """
    Compute multi‑query attention (GQA with n_kv=1).

    Parameters
    ----------
    Q : list[list[list[float]]]
        Queries of shape (B, N_q, d_k).
    K : list[list[list[float]]]
        Keys of shape (B, N_k, d_k).
    V : list[list[list[float]]]
        Values of shape (B, N_v, d_v).

    Returns
    -------
    list[list[list[float]]]
        Attention output of shape (B, N_q, d_v).
    """
    B = len(Q)
    N_q = len(Q[0])
    d_k = len(Q[0][0])

    N_k = len(K[0])
    d_v = len(V[0][0])

    scale = math.sqrt(d_k)
    out = [[[0.0] * d_v for _ in range(N_q)] for _ in range(B)]

    for b in range(B):
        for q in range(N_q):
            max_score = -float('inf')
            scores = [0.0] * N_k
            for k in range(N_k):
                dot = 0.0
                for d in range(d_k):
                    dot += Q[b][q][d] * K[b][k][d]
                score = dot / scale
                scores[k] = score
                if score > max_score:
                    max_score = score

            sum_exp = 0.0
            weights = [0.0] * N_k
            for k in range(N_k):
                w = math.exp(scores[k] - max_score)
                weights[k] = w
                sum_exp += w

            for v_dim in range(d_v):
                val = 0.0
                for k in range(N_k):
                    val += (weights[k] / sum_exp) * V[b][k][v_dim]
                out[b][q][v_dim] = val

    return out
