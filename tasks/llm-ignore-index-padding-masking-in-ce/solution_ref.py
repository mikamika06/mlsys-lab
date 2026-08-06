import math

def masked_cross_entropy(logits: list[list[float]],
                         targets: list[int],
                         ignore_index: int = -100) -> float:
    n_samples = len(logits)
    if n_samples == 0:
        return 0.0
    n_classes = len(logits[0])

    total_loss = 0.0
    valid_count = 0

    for i in range(n_samples):
        target = targets[i]
        if target == ignore_index:
            continue

        max_val = float(logits[i][0])
        for j in range(1, n_classes):
            val = float(logits[i][j])
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        for j in range(n_classes):
            sum_exp += math.exp(float(logits[i][j]) - max_val)

        log_prob = float(logits[i][target]) - max_val - math.log(sum_exp)
        total_loss += -log_prob
        valid_count += 1

    if valid_count == 0:
        return 0.0

    return float(total_loss / valid_count)
