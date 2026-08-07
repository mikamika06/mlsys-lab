def compute_net_savings(hit_rate, total_requests, compute_cost_per_req, memory_cost_per_entry):
    raise NotImplementedError


def compute_breakeven_hit_rate(compute_cost_per_req, total_requests, total_memory_cost):
    raise NotImplementedError


def evaluate_cache_viability(capacity, trace, compute_cost_per_req, memory_cost_per_entry):
    raise NotImplementedError
