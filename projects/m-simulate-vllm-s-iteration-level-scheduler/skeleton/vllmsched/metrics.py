def calculate_throughput(completed_requests, total_steps):
    raise NotImplementedError


def measure_concurrency_sweep(
    requests_generator, concurrency_levels, num_blocks, block_size, max_tokens
):
    raise NotImplementedError
