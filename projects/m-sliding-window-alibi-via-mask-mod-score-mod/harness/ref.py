import numpy as np

def alibi_score_mod(score, h, q, kv, num_heads):
    slope = 2 ** (-8.0 * (h + 1) / num_heads)
    return score - slope * (q - kv)

def sliding_window_mask_mod(b, h, q, kv, window):
    return (q >= kv) & ((q - kv) < window)

def compute_block_mask_indices(seq_len, window, block_size):
    num_q_blocks = (seq_len + block_size - 1) // block_size
    num_kv_blocks = (seq_len + block_size - 1) // block_size

    total_blocks = num_q_blocks * num_kv_blocks
    empty_blocks = 0

    kv_num_blocks = np.zeros(num_q_blocks, dtype=np.int32)
    kv_indices_list = []

    for q_b in range(num_q_blocks):
        q_min = q_b * block_size
        q_max = min(seq_len - 1, q_min + block_size - 1)

        valid_kv = []
        for kv_b in range(num_kv_blocks):
            kv_min = kv_b * block_size
            kv_max = min(seq_len - 1, kv_min + block_size - 1)

            if kv_min <= q_max and kv_max > q_min - window:
                valid_kv.append(kv_b)
            else:
                empty_blocks += 1

        kv_num_blocks[q_b] = len(valid_kv)
        kv_indices_list.append(valid_kv)

    max_kv = int(np.max(kv_num_blocks)) if num_q_blocks > 0 else 0
    kv_indices = np.zeros((num_q_blocks, max_kv), dtype=np.int32)

    for q_b, indices in enumerate(kv_indices_list):
        kv_indices[q_b, :len(indices)] = indices

    sparsity = empty_blocks / total_blocks if total_blocks > 0 else 0.0
    return float(sparsity), kv_num_blocks, kv_indices
