def compute_allocation_schedule(max_c, block_size):
    blocks = []
    current = 0
    while current < max_c:
        nxt = min(current + block_size, max_c)
        blocks.append((current, nxt))
        current = nxt
    return blocks
