import math

def cross_entropy_from_logits(logits: list[list[float]], targets: list[int]) -> float:
    n_samples = len(logits)
    n_classes = len(logits[0])
    ce_sum = 0.0

    for i in range(n_samples):
        row = logits[i]

        m = row[0]
        for j in range(1, n_classes):
            val = row[j]
            if val > m:
                m = val

        sum_exp = 0.0
        for j in range(n_classes):
            sum_exp += math.exp(row[j] - m)

        target_idx = targets[i]
        log_prob_target = row[target_idx] - m - math.log(sum_exp)
        ce_sum += -log_prob_target

    return ce_sum / n_samples
