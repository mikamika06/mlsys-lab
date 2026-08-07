def generate_zigzag_assignments(num_tokens, world_size):
    block_size = num_tokens // (world_size * 2)
    assignments = [[] for _ in range(world_size)]
    for step in range(world_size * 2):
        if step % 2 == 0:
            rank = step // 2
        else:
            rank = world_size - 1 - (step // 2)
        start = step * block_size
        end = start + block_size
        assignments[rank].extend(list(range(start, end)))
    return assignments
