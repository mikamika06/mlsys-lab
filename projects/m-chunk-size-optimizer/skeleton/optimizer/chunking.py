def calculate_prefix_savings(trace: list[list[int]], chunk_size: int) -> int:
    """Calculate total tokens saved by sharing exactly matching full chunks."""
    raise NotImplementedError

def benchmark_serving_frontier(trace: list[list[int]], chunk_size: int, saved_tokens: int) -> float:
    """Simulate a serving frontier metric penalizing compute, fragmentation, and overhead."""
    raise NotImplementedError

def optimize_chunk_size(trace: list[list[int]], candidate_sizes: list[int]) -> int:
    """Return the index of the candidate size that yields the lowest serving frontier cost."""
    raise NotImplementedError
