import math


def sliding_window_document_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    doc_ids: list[int],
    window: int,
) -> tuple[list[list[float]], list[list[bool]]]:
    n = len(Q)
    d = len(Q[0])
    d_v = len(V[0])

    mask = [[False] * n for _ in range(n)]
    logits = [[0.0] * n for _ in range(n)]
    scaled_d = math.sqrt(d)

    for i in range(n):
        for j in range(n):
            cond = (j <= i) and ((i - j) < window) and (doc_ids[i] == doc_ids[j])
            mask[i][j] = cond
            if cond:
                dot_val = 0.0
                for k_dim in range(d):
                    dot_val += Q[i][k_dim] * K[j][k_dim]
                logits[i][j] = dot_val / scaled_d
            else:
                logits[i][j] = -float("inf")

    probs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        max_val = -float("inf")
        for j in range(n):
            if mask[i][j]:
                if logits[i][j] > max_val:
                    max_val = logits[i][j]

        row_sum = 0.0
        for j in range(n):
            if mask[i][j]:
                val = math.exp(logits[i][j] - max_val)
                probs[i][j] = val
                row_sum += val

        if row_sum > 0.0:
            for j in range(n):
                if mask[i][j]:
                    probs[i][j] /= row_sum

    out = [[0.0] * d_v for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p = probs[i][j]
            if p != 0.0:
                for k_dim in range(d_v):
                    out[i][k_dim] += p * V[j][k_dim]

    return out, mask
