import math


def causal_self_attention(
    Q: list[list[float]], K: list[list[float]], V: list[list[float]]
) -> list[list[float]]:
    """Causal scaled dot-product self-attention.

    Q, K, V: (n, d). Row i may only attend to keys/values at position
    <= i. Masking is applied to the LOGITS (score[i, j] = -inf for j > i)
    BEFORE softmax, so every row's probabilities still sum to 1 over the
    positions it is allowed to see. Returns (n, d).
    """
    n = len(Q)
    d = len(Q[0])
    sqrt_d = math.sqrt(d)

    out = []
    for i in range(n):
        scores = []
        for j in range(n):
            if j > i:
                scores.append(float("-inf"))
            else:
                dot = sum(Q[i][k] * K[j][k] for k in range(d))
                scores.append(dot / sqrt_d)

        max_score = max(scores[: i + 1])

        exps = []
        exp_sum = 0.0
        for j in range(n):
            if j > i:
                exps.append(0.0)
            else:
                val = math.exp(scores[j] - max_score)
                exps.append(val)
                exp_sum += val

        probs = [val / exp_sum for val in exps]

        row_out = []
        for k in range(d):
            val = sum(probs[j] * V[j][k] for j in range(i + 1))
            row_out.append(val)
        out.append(row_out)

    return out
