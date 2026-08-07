import numpy as np


def ring_attention(q, k, v, rank, world_size):
    seq_len, head_dim = q.shape[0], q.shape[1]
    scale = 1.0 / np.sqrt(head_dim)

    local_k = k.copy()
    local_v = v.copy()

    acc_scores = np.full((seq_len, 1), -np.inf)
    acc_out = np.zeros((seq_len, head_dim))

    for step in range(world_size):
        target_rank = (rank - step) % world_size

        scores = np.matmul(q, local_k.T) * scale

        if step == 0:
            row_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - row_max)
            sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
            acc_out = np.matmul(exp_scores, local_v)
            acc_scores = row_max + np.log(sum_exp + 1e-12)
        else:
            row_max = np.max(scores, axis=-1, keepdims=True)
            new_max = np.maximum(acc_scores, row_max)

            old_scale = np.exp(acc_scores - new_max)
            new_scale = np.exp(row_max - new_max)

            acc_out = acc_out * old_scale + np.matmul(np.exp(scores - row_max), local_v) * new_scale
            sum_old = np.exp(acc_scores - new_max)
            sum_new = np.exp(row_max - new_max) * np.sum(np.exp(scores - row_max), axis=-1, keepdims=True)
            acc_scores = new_max + np.log(sum_old * np.sum(np.exp(acc_scores - acc_scores), axis=-1, keepdims=True) + sum_new + 1e-12)

        local_k = np.roll(local_k, shift=seq_len, axis=0)
        local_v = np.roll(local_v, shift=seq_len, axis=0)

    return acc_out
