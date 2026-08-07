import math


def packed_attention_with_reset_mask(Q: list[list[float]], K: list[list[float]], V: list[list[float]], segment_ids: list[int]) -> list[list[float]]:
    """Causal self-attention over multiple documents PACKED into one
    training sequence, with the mask RESET at every segment boundary.

    Q, K, V: (n, d). segment_ids: (n,) int list; segment_ids[i] is the
    segment/document index token i belongs to (e.g. [0,0,0,1,1,2,2,2,2] for
    three packed documents of length 3, 2, 4).

    Row i may attend to column j iff j <= i (causal) AND segment_ids[j] ==
    segment_ids[i] (same document -- the mask resets, exactly like resetting
    position ids at each packed-document boundary). Returns (n, d).
    """
    n = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    scores = []
    for i in range(n):
        row_scores = []
        for j in range(n):
            if j <= i and segment_ids[j] == segment_ids[i]:
                dot = 0.0
                for k in range(d):
                    dot += Q[i][k] * K[j][k]
                row_scores.append(dot * scale)
            else:
                row_scores.append(-float("inf"))
        scores.append(row_scores)

    output = []
    for i in range(n):
        max_val = -float("inf")
        for j in range(n):
            if scores[i][j] > max_val:
                max_val = scores[i][j]

        exps = []
        sum_exp = 0.0
        for j in range(n):
            if scores[i][j] == -float("inf"):
                exps.append(0.0)
            else:
                val = math.exp(scores[i][j] - max_val)
                exps.append(val)
                sum_exp += val

        probs = [val / sum_exp for val in exps]

        row_out = []
        for k in range(d):
            val = 0.0
            for j in range(n):
                val += probs[j] * V[j][k]
            row_out.append(val)
        output.append(row_out)

    return output
