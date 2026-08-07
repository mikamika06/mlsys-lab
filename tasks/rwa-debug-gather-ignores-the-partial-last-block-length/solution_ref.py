import math


def gathered_attention(
    k_phys: list[list[list[float]]],
    v_phys: list[list[list[float]]],
    block_table: list[int],
    seq_len: int,
    q: list[float],
) -> list[float]:
    """Gather logical KV from a block table, truncate to the true seq_len,
    and compute single-query scaled dot-product attention."""
    H = len(q)

    k_logical = []
    v_logical = []
    for b_idx in block_table:
        for row_k, row_v in zip(k_phys[b_idx], v_phys[b_idx]):
            k_logical.append(list(row_k))
            v_logical.append(list(row_v))

    k_logical = k_logical[:seq_len]
    v_logical = v_logical[:seq_len]

    scale = math.sqrt(H)
    scores = []
    for row in k_logical:
        dot = sum(a * b for a, b in zip(row, q))
        scores.append(dot / scale)

    max_score = max(scores)
    weights = [math.exp(s - max_score) for s in scores]
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]

    out = [0.0] * H
    for w, row in zip(weights, v_logical):
        for j in range(H):
            out[j] += w * row[j]

    return out
