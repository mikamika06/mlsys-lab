def gqa_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], g: int) -> list[list[float]]:
    """
    Plain Python implementation of grouped‑query attention.
    """
    n_q = len(Q)
    d = len(Q[0])
    out = [[0.0 for _ in range(d)] for _ in range(n_q)]
    for i in range(n_q):
        j = i // g
        score = 0.0
        for k in range(d):
            score += Q[i][k] * K[j][k]
        for k in range(d):
            out[i][k] = score * V[j][k]
    return out
