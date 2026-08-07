import numpy as np


def ring_attention(q_chunks, k_chunks, v_chunks):
    world_size = len(q_chunks)
    outputs = []
    for rank in range(world_size):
        q = q_chunks[rank]
        scale = 1.0 / np.sqrt(q.shape[-1])

        running_max = np.full((*q.shape[:-1], 1), -np.inf, dtype=q.dtype)
        running_sum = np.zeros((*q.shape[:-1], 1), dtype=q.dtype)
        running_out = np.zeros_like(q)

        curr_k = list(k_chunks)
        curr_v = list(v_chunks)

        for step in range(world_size):
            send_idx = (rank - step) % world_size
            k_block = curr_k[send_idx]
            v_block = curr_v[send_idx]

            scores = np.matmul(q, np.swapaxes(k_block, -2, -1)) * scale
            block_max = np.max(scores, axis=-1, keepdims=True)

            new_max = np.maximum(running_max, block_max)
            exp_old = np.exp(running_max - new_max)
            exp_new = np.exp(block_max - new_max)

            exp_scores = np.exp(scores - new_max)
            block_sum = np.sum(exp_scores, axis=-1, keepdims=True)

            updated_sum = running_sum * exp_old + block_sum * exp_new

            inter = running_out * running_sum * exp_old + np.matmul(exp_scores, v_block) * exp_new
            running_out = inter / updated_sum
            running_max = new_max
            running_sum = updated_sum

        outputs.append(running_out)
    return outputs
