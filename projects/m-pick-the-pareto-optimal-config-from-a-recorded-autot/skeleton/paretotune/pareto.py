def is_dominated(candidate, population):
    """Check if candidate metrics are dominated by any record in population."""
    raise NotImplementedError


def find_pareto_frontier(records):
    """Filter records to return only non-dominated entries based on latency, memory, and registers."""
    raise NotImplementedError
