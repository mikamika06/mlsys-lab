import numpy as np


def merge_lse_pair(out_a, max_a, sum_a, out_b, max_b, sum_b):
    new_max = np.maximum(max_a, max_b)
    alpha = np.exp(max_a - new_max)
    beta = np.exp(max_b - new_max)
    new_sum = alpha * sum_a + beta * sum_b
    out_a_scaled = out_a * (alpha[..., None] * sum_a[..., None])
    out_b_scaled = out_b * (beta[..., None] * sum_b[..., None])
    new_out = (out_a_scaled + out_b_scaled) / np.maximum(new_sum[..., None], 1e-30)
    return new_out, new_max, new_sum


def merge_partial_outputs(partial_results):
    if not partial_results:
        raise ValueError("Empty partial results")
    out, m, l = partial_results[0]
    for next_out, next_m, next_l in partial_results[1:]:
        out, m, l = merge_lse_pair(out, m, l, next_out, next_m, next_l)
    return out, m, l
