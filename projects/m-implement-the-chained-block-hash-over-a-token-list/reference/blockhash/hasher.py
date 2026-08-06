def compute_chained_hashes(tokens, block_size=16):
    hashes = []
    current_hash = 0
    for i in range(0, len(tokens), block_size):
        block = tokens[i:i + block_size]
        padded = list(block) + [0] * (block_size - len(block))
        h = current_hash
        for t in padded:
            h = (h * 31 + t) & 0xFFFFFFFFFFFFFFFF
        current_hash = h
        hashes.append(current_hash)
    return hashes
