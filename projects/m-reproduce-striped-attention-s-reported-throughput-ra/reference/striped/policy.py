def assign_blocks(num_blocks, world_size):
    assignment = [[] for _ in range(world_size)]
    for i in range(num_blocks):
        rank = i % world_size
        assignment[rank].append(i)
    return assignment
