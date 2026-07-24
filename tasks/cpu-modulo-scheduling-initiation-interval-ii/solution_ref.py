def compute_initiation_interval(resource_bound: int, recurrence_bound: int) -> int:
    return max(resource_bound, recurrence_bound)
