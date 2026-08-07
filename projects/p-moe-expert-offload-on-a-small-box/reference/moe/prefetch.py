def prefetch_plan(routing_decisions, cache, bandwidth):
    planned = []
    for step in routing_decisions:
        needed = step.get("next_experts", [])
        for exp in needed:
            if exp not in cache.cache:
                planned.append(exp)
    return planned
