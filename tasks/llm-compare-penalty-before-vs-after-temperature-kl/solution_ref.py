import math
from typing import Callable


def compare_penalty_temperature(
    logits: list[list[float]],
    penalty_fn: Callable[[list[list[float]]], list[list[float]]],
    temperature: float,
) -> float:
    def softmax_2d(arr: list[list[float]]) -> list[list[float]]:
        res = []
        for row in arr:
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exp = sum(exps)
            res.append([e / sum_exp for e in exps])
        return res

    # before temp
    penalized_before = penalty_fn(logits)
    scaled_before = [[x / temperature for x in row] for row in penalized_before]
    probs_before = softmax_2d(scaled_before)

    # after temp
    scaled_after = [[x / temperature for x in row] for row in logits]
    penalized_after = penalty_fn(scaled_after)
    probs_after = softmax_2d(penalized_after)

    # mean KL divergence
    total_kl = 0.0
    batch_size = len(logits)

    for r in range(batch_size):
        row_kl = 0.0
        p_row = probs_before[r]
        q_row = probs_after[r]
        for p, q in zip(p_row, q_row):
            row_kl += p * (math.log(p + 1e-12) - math.log(q + 1e-12))
        total_kl += row_kl

    return total_kl / batch_size
