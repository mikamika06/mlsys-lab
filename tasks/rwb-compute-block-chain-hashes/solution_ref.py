import numpy as np

def compute_block_hashes(tokens, block_size, salt):
    tokens = np.asarray(tokens, dtype=np.uint64)
    n = tokens.size
    num_blocks = (n + block_size - 1) // block_size
    hashes = np.empty(num_blocks, dtype=np.uint64)
    h_prev = np.uint64(salt)
    prime = np.uint64(1099511628211)
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        h = np.uint64(0)
        # fold previous hash
        h ^= h_prev
        h *= prime
        for t in tokens[start:end]:
            h ^= t
            h *= prime
        hashes[i] = h
        h_prev = h
    return hashes
