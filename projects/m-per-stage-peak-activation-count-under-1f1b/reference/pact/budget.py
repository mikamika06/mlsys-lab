def max_microbatches(num_stages: int, memory_budget_bytes: int, activation_bytes_per_mb: int) -> int:
    max_allowed = memory_budget_bytes // activation_bytes_per_mb
    return max(0, max_allowed)
