import math


def fused_cross_entropy(logits: list[list[float]], targets: list[int]) -> list[float]:
    """Per-example cross-entropy loss ell_i = logsumexp(logits[i]) - logits[i, targets[i]],
    computed via the numerically-stable log-sum-exp trick."""
    num_rows = len(logits)
    num_cols = len(logits[0])

    result = [0.0] * num_rows

    for i in range(num_rows):
        row = logits[i]
        max_val = row[0]
        for j in range(1, num_cols):
            if row[j] > max_val:
                max_val = row[j]

        sum_exp = 0.0
        for j in range(num_cols):
            sum_exp += math.exp(row[j] - max_val)

        lse = max_val + math.log(sum_exp)
        tgt_logit = row[targets[i]]
        result[i] = lse - tgt_logit

    return result
