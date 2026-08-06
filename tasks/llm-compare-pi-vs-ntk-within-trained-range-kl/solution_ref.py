import math

def compare_pi_ntk(base_logits: list[list[float]],
                   pi_logits: list[list[float]],
                   ntk_logits: list[list[float]]) -> tuple[float, float]:
    """
    Compute the mean KL divergence between each of two sets of logits and a base set.
    The implementation uses pure Python with explicit loops and math functions.
    """
    def _mean_kl(a: list[list[float]], b: list[list[float]]) -> float:
        kl_sum = 0.0
        n = len(a)
        for row_a, row_b in zip(a, b):
            # Softmax for row_a
            max_a = max(row_a)
            exp_a = [math.exp(x - max_a) for x in row_a]
            sum_exp_a = sum(exp_a)
            p = [val / sum_exp_a for val in exp_a]

            # Softmax for row_b
            max_b = max(row_b)
            exp_b = [math.exp(x - max_b) for x in row_b]
            sum_exp_b = sum(exp_b)
            q = [val / sum_exp_b for val in exp_b]

            # KL divergence for this row
            row_kl = 0.0
            for pi_val, qi_val in zip(p, q):
                log_p = math.log(pi_val + 1e-12)
                log_q = math.log(qi_val + 1e-12)
                row_kl += pi_val * (log_p - log_q)
            kl_sum += row_kl

        return kl_sum / n

    pi_kl = _mean_kl(base_logits, pi_logits)
    ntk_kl = _mean_kl(base_logits, ntk_logits)

    return pi_kl, ntk_kl
