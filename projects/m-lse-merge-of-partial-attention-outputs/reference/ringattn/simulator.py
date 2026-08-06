import numpy as np
from ringattn.lse import merge_lse_pair


def _compute_block_attn(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    max_val = np.max(scores, axis=-1)
    exp_scores = np.exp(scores - max_val[..., None])
    if mask is not None:
        exp_scores = np.where(mask, exp_scores, 0.0)
    sum_val = np.sum(exp_scores, axis=-1)
    out = np.matmul(exp_scores, v) / np.maximum(sum_val[..., None], 1e-30)
    return out, max_val, sum_val


def run_ring_attention(q_blocks, k_blocks, v_blocks, is_causal=False):
    num_ranks = len(q_blocks)
    seq_len_per_rank = q_blocks[0].shape[0]
    head_dim = q_blocks[0].shape[-1]

    acc_out = [np.zeros((seq_len_per_rank, head_dim)) for _ in range(num_ranks)]
    acc_max = [np.full((seq_len_per_rank,), -1e9) for _ in range(num_ranks)]
    acc_sum = [np.zeros((seq_len_per_rank,)) for _ in range(num_ranks)]

    k_ring = list(k_blocks)
    v_ring = list(v_blocks)

    for step in range(num_ranks):
        for r in range(num_ranks):
            kv_idx = (r - step) % num_ranks
            q_b = q_blocks[r]
            k_b = k_ring[r]
            v_b = v_ring[r]

            mask = None
            if is_causal:
                if r < kv_idx:
                    continue
                elif r == kv_idx:
                    q_indices = np.arange(seq_len_per_rank)[:, None]
                    k_indices = np.arange(seq_len_per_rank)[None, :]
                    mask = q_indices >= k_indices

            out, max_val, sum_val = _compute_block_attn(q_b, k_b, v_b, mask=mask)
            acc_out[r], acc_max[r], acc_sum[r] = merge_lse_pair(
                acc_out[r], acc_max[r], acc_sum[r], out, max_val, sum_val
            )

        k_ring = [k_ring[(i - 1) % num_ranks] for i in range(num_ranks)]
        v_ring = [v_ring[(i - 1) % num_ranks] for i in range(num_ranks)]

    return acc_out
