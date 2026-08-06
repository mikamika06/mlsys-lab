from blockhash.hasher import compute_chained_hashes

def count_reusable_blocks(tokens_a, tokens_b, block_size=16):
    hashes_a = compute_chained_hashes(tokens_a, block_size)
    hashes_b = compute_chained_hashes(tokens_b, block_size)
    count = 0
    for ha, hb in zip(hashes_a, hashes_b):
        if ha == hb:
            count += 1
        else:
            break
    return count
