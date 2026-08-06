def calculate_prefix_savings(trace: list[list[int]], chunk_size: int) -> int:
    """Calculate total tokens saved by sharing exactly matching full chunks."""
    saved = 0
    seen = set()
    for req in trace:
        for i in range(0, len(req), chunk_size):
            chunk = tuple(req[i:i+chunk_size])
            if len(chunk) == chunk_size:
                if chunk in seen:
                    saved += chunk_size
                else:
                    seen.add(chunk)
    return saved

def benchmark_serving_frontier(trace: list[list[int]], chunk_size: int, saved_tokens: int) -> float:
    """Simulate a serving frontier metric penalizing compute, fragmentation, and overhead."""
    total_tokens = sum(len(req) for req in trace)
    compute = (total_tokens - saved_tokens) * 0.5
    frag = (chunk_size * len(trace)) * 0.1
    overhead = (total_tokens / chunk_size) * 1.5
    return float(compute + frag + overhead)

def optimize_chunk_size(trace: list[list[int]], candidate_sizes: list[int]) -> int:
    """Return the index of the candidate size that yields the lowest serving frontier cost."""
    best_idx = 0
    min_cost = float('inf')
    for i, size in enumerate(candidate_sizes):
        saved = calculate_prefix_savings(trace, size)
        cost = benchmark_serving_frontier(trace, size, saved)
        if cost < min_cost:
            min_cost = cost
            best_idx = i
    return best_idx
