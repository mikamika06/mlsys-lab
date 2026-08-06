import math

def label_smoothed_cross_entropy(logits: list[list[float]],
                                 targets: list[int],
                                 eps: float = 0.1) -> float:
    """
    Compute the average label‑smoothed cross‑entropy loss.
    """
    N = len(logits)
    K = len(logits[0])
    eps_over_K = eps / K
    one_minus_eps = 1.0 - eps

    total_loss = 0.0
    for i in range(N):
        target_i = targets[i]

        max_logit = float(logits[i][0])
        for k in range(1, K):
            val = float(logits[i][k])
            if val > max_logit:
                max_logit = val

        sum_exp = 0.0
        for k in range(K):
            sum_exp += math.exp(float(logits[i][k]) - max_logit)

        log_sum_exp = math.log(sum_exp)

        sample_loss_sum = 0.0
        for k in range(K):
            y_s = (one_minus_eps + eps_over_K) if k == target_i else eps_over_K
            log_sm = (float(logits[i][k]) - max_logit) - log_sum_exp
            sample_loss_sum += y_s * log_sm

        total_loss += -sample_loss_sum

    return float(total_loss / N)
