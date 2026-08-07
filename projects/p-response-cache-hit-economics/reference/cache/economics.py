"""Economic benefit and breakeven models for LLM cache serving."""


def compute_net_savings(hit_rate, total_requests, compute_cost_per_req, memory_cost_per_entry):
    hits = hit_rate * total_requests
    compute_saved = hits * compute_cost_per_req
    net_savings = compute_saved - memory_cost_per_entry
    return {
        "compute_saved": compute_saved,
        "net_savings": net_savings,
        "is_profitable": net_savings > 0,
    }


def compute_breakeven_hit_rate(compute_cost_per_req, total_requests, total_memory_cost):
    if total_requests <= 0 or compute_cost_per_req <= 0:
        return 1.0
    required_hits = total_memory_cost / compute_cost_per_req
    breakeven_rate = required_hits / total_requests
    return min(1.0, max(0.0, breakeven_rate))


def evaluate_cache_viability(capacity, trace, compute_cost_per_req, memory_cost_per_entry):
    from cache.eviction import CacheSimulator

    sim = CacheSimulator(capacity=capacity, policy="lru")
    for req in trace:
        sim.access(req["key"], compute_cost=req.get("cost", compute_cost_per_req))

    stats = sim.get_stats()
    total_reqs = len(trace)
    total_memory_cost = capacity * memory_cost_per_entry
    hits = stats["hits"]
    hit_rate = stats["hit_rate"]

    compute_saved = hits * compute_cost_per_req
    net_savings = compute_saved - total_memory_cost
    roi = (net_savings / total_memory_cost) if total_memory_cost > 0 else 0.0
    breakeven_rate = compute_breakeven_hit_rate(compute_cost_per_req, total_reqs, total_memory_cost)

    return {
        "capacity": capacity,
        "hit_rate": hit_rate,
        "compute_saved": compute_saved,
        "memory_cost": total_memory_cost,
        "net_savings": net_savings,
        "roi": roi,
        "breakeven_hit_rate": breakeven_rate,
        "should_enable": net_savings > 0 and hit_rate >= breakeven_rate,
    }
