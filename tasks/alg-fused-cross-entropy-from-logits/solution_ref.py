def fused_cross_entropy(logits: list[list[float]], targets: list[int]) -> float:
    import math

    N = len(logits)
    C = len(logits[0]) if N > 0 else 0

    total_ce = 0.0

    for i in range(N):
        row = logits[i]
        row_max = row[0]
        for j in range(1, C):
            if row[j] > row_max:
                row_max = row[j]

        exp_sum = 0.0
        for j in range(C):
            exp_sum += math.exp(row[j] - row_max)

        logsumexp = math.log(exp_sum) + row_max
        target_val = row[targets[i]]

        ce = - (target_val - logsumexp)
        total_ce += ce

    return float(total_ce / N)
