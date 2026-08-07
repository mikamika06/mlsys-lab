from vllmsched.scheduler import Scheduler


def calculate_throughput(completed_requests, total_steps):
    if total_steps == 0:
        return 0.0
    total_tokens = sum(r.prompt_len + r.output_len for r in completed_requests)
    return total_tokens / total_steps


def measure_concurrency_sweep(
    requests_generator, concurrency_levels, num_blocks, block_size, max_tokens
):
    results = {}
    for conc in concurrency_levels:
        requests = requests_generator(conc)
        sched = Scheduler(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_batched_tokens=max_tokens,
        )
        completed = sched.run_simulation(requests)
        max_step = max((r.completion_time for r in completed), default=0)
        throughput = calculate_throughput(completed, max_step)
        results[conc] = throughput
    return results
