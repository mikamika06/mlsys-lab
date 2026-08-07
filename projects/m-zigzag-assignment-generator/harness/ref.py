CONFIGS = [
    {"num_tokens": 64, "world_size": 4},
    {"num_tokens": 128, "world_size": 8},
    {"num_tokens": 256, "world_size": 2},
]

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

def compute_comm_volume(num_tokens, world_size, head_dim, dtype_size):
    tokens_per_rank = num_tokens // world_size
    bytes_per_token = head_dim * dtype_size
    steps = world_size - 1
    return tokens_per_rank * bytes_per_token * steps

def check_overlap_feasibility(comm_time, compute_time, overlap_efficiency):
    effective_comm = comm_time * (1.0 - overlap_efficiency)
    return effective_comm <= compute_time
