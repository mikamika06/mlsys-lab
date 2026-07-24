import numpy as np

def mark_kv_blocks(seq_lengths: np.ndarray, block_size: int) -> np.ndarray:
    seq_lengths = np.asarray(seq_lengths, dtype=int)
    n_blocks_per_seq = (seq_lengths + block_size - 1) // block_size
    total_blocks = int(n_blocks_per_seq.sum())
    labels = np.empty(total_blocks, dtype=bool)
    idx = 0
    for L, nb in zip(seq_lengths, n_blocks_per_seq):
        full_blocks = nb if L % block_size == 0 else nb - 1
        labels[idx:idx+full_blocks] = True
        idx += full_blocks
        if nb > full_blocks:
            labels[idx] = False
            idx += 1
    return labels
