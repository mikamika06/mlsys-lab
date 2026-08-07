import numpy as np


def ulysses_attention(q, k, v, rank, world_size, num_heads):
    batch, seq_len_per_rank, head_dim = q.shape[0], q.shape[1], q.shape[2]

    heads_per_rank = num_heads // world_size
    q_reshaped = q.reshape(batch, seq_len_per_rank, world_size, heads_per_rank, head_dim)
    q_transposed = np.transpose(q_reshaped, (0, 3, 1, 2, 4))

    k_reshaped = k.reshape(batch, seq_len_per_rank, world_size, heads_per_rank, head_dim)
    k_transposed = np.transpose(k_reshaped, (0, 3, 1, 2, 4))

    v_reshaped = v.reshape(batch, seq_len_per_rank, world_size, heads_per_rank, head_dim)
    v_transposed = np.transpose(v_reshaped, (0, 3, 1, 2, 4))

    q_local = q_transposed[:, rank]
    k_local = k_transposed[:, rank]
    v_local = v_transposed[:, rank]

    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q_local, np.swapaxes(k_local, -1, -2)) * scale

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn_out = np.matmul(exp_scores / (sum_exp + 1e-12), v_local)

    out_reshaped = np.expand_dims(attn_out, axis=3)
    out_broadcasted = np.tile(out_reshaped, (1, 1, 1, world_size, 1))
    out_transposed = np.transpose(out_broadcasted, (0, 2, 1, 3, 4))
    final_out = out_transposed.reshape(batch, seq_len_per_rank, heads_per_rank * head_dim)

    return final_out
