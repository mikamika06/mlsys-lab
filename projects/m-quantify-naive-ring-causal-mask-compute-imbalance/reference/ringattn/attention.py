import numpy as np


def distributed_ring_attention_step(rank, world_size, q, k, v, block_size):
    num_blocks = q.shape[0] // block_size
    blocks_per_rank = num_blocks // world_size
    start_block = rank * blocks_per_rank
    end_block = (rank + 1) * blocks_per_rank

    q_local = q[start_block * block_size : end_block * block_size]

    out_local = np.zeros_like(q_local)
    scale = 1.0 / np.sqrt(q.shape[-1])

    for step in range(world_size):
        k_target_rank = (rank - step) % world_size
        k_start = k_target_rank * blocks_per_rank * block_size
        k_end = (k_target_rank + 1) * blocks_per_rank * block_size
        k_chunk = k[k_start:k_end]
        v_chunk = v[k_start:k_end]

        for i in range(blocks_per_rank):
            global_q_idx = start_block + i
            q_blk = q_local[i * block_size : (i + 1) * block_size]
            for j in range(blocks_per_rank):
                global_k_idx = k_target_rank * blocks_per_rank + j
                if global_k_idx <= global_q_idx:
                    k_blk = k_chunk[j * block_size : (j + 1) * block_size]
                    v_blk = v_chunk[j * block_size : (j + 1) * block_size]
                    scores = np.matmul(q_blk, k_blk.T) * scale
                    max_val = np.max(scores, axis=-1, keepdims=True)
                    exp_scores = np.exp(scores - max_val)
                    out_local[i * block_size : (i + 1) * block_size] += np.matmul(exp_scores, v_blk)

    return out_local
