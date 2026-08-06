import numpy as np

def mark_kv_blocks(seq_lengths: np.ndarray, block_size: int) -> np.ndarray:
    seq_lengths = np.asarray(seq_lengths, dtype=int)
    n_blocks_per_seq = np.empty(len(seq_lengths), dtype=int)
    for i in range(len(seq_lengths)):
        L = int(seq_lengths[i])
        n_blocks_per_seq[i] = (L + block_size - 1) // block_size
    total_blocks = 0
    for i in range(len(n_blocks_per_seq)):
        total_blocks += int(n_blocks_per_seq[i])
    labels = np.empty(total_blocks, dtype=bool)
    idx = 0
    for i in range(len(seq_lengths)):
        L = int(seq_lengths[i])
        nb = int(n_blocks_per_seq[i])
        full_blocks = nb if L % block_size == 0 else nb - 1
        for j in range(full_blocks):
            labels[idx] = True
            idx += 1
        if nb > full_blocks:
            labels[idx] = False
            idx += 1
    return labels
