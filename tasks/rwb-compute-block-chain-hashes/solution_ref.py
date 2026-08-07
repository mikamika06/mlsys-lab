def compute_block_hashes(tokens, block_size, salt):
    n = len(tokens)
    num_blocks = (n + block_size - 1) // block_size
    hashes = []
    h_prev = salt & 0xFFFFFFFFFFFFFFFF
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        h = 0
        # fold previous hash
        h ^= h_prev
        h = (h * prime) & mask
        for t in tokens[start:end]:
            h ^= t
            h = (h * prime) & mask
        hashes.append(h)
        h_prev = h
    return hashes
