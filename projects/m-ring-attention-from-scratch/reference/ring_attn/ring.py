import numpy as np


def ring_attention(q_chunks, k_chunks, v_chunks):
    world_size = len(q_chunks)
    outputs = []
    for rank in range(world_size):
        q = q_chunks[rank]
        running_max = None
        running_sum = None
        running_out = None
        for step in range(world_size):
            k_idx = (rank - step) % world_size
            k = k_chunks[k_idx]
            v = v_chunks[k_idx]
            scale = 1.0 / np.sqrt(q.shape[-1])
            scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
            block_max = np.max(scores, axis=-1, keepdims=True)
            block_exp = np.exp(scores - block_max)
            block_sum = np.sum(block_exp, axis=-1, keepdims=True)
            if step == 0:
                running_max = block_max
                running_sum = block_sum
                running_out = np.matmul(block_exp, v)
            else:
                new_max = np.maximum(running_max, block_max)
                c_old = np.exp(running_max - new_max)
                c_new = np.exp(block_max - new_max)
                running_sum = running_sum * c_old + block_sum * c_new
                running_out = running_out * c_old + np.matmul(block_exp, v) * c_new
                running_max = new_max
        outputs.append(running_out / running_sum)
    return outputs
