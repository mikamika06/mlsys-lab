import numpy as np


def generate_lse_test_case(seq_len=32, head_dim=64, num_partials=3, seed=42):
    rng = np.random.RandomState(seed)
    partials = []
    for _ in range(num_partials):
        out = rng.randn(seq_len, head_dim)
        m = rng.randn(seq_len)
        l = rng.uniform(0.5, 5.0, size=(seq_len,))
        partials.append((out, m, l))
    return partials


def ref_merge_lse_pair(out_a, max_a, sum_a, out_b, max_b, sum_b):
    new_max = np.maximum(max_a, max_b)
    alpha = np.exp(max_a - new_max)
    beta = np.exp(max_b - new_max)
    new_sum = alpha * sum_a + beta * sum_b
    out_a_scaled = out_a * (alpha[..., None] * sum_a[..., None])
    out_b_scaled = out_b * (beta[..., None] * sum_b[..., None])
    new_out = (out_a_scaled + out_b_scaled) / np.maximum(new_sum[..., None], 1e-30)
    return new_out, new_max, new_sum


def ref_merge_partial_outputs(partial_results):
    out, m, l = partial_results[0]
    for next_out, next_m, next_l in partial_results[1:]:
        out, m, l = ref_merge_lse_pair(out, m, l, next_out, next_m, next_l)
    return out, m, l


def ref_compute_causal_load_imbalance(world_size, num_blocks):
    total_blocks = world_size * num_blocks
    rank_work = np.zeros(world_size, dtype=np.int64)
    for r in range(world_size):
        q_start = r * num_blocks
        q_end = (r + 1) * num_blocks
        for b in range(total_blocks):
            if b < q_start:
                rank_work[r] += num_blocks
            elif b < q_end:
                n_full = b - q_start
                rank_work[r] += n_full
    max_work = float(np.max(rank_work))
    avg_work = float(np.mean(rank_work))
    ratio = max_work / max(avg_work, 1e-9)
    return {"rank_work": rank_work.tolist(), "imbalance_ratio": ratio}
