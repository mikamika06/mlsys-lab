def simulate_scheduler(prompts, decodes, token_budget):
    steps = 0
    active_prompts = list(prompts)
    while active_prompts:
        used = decodes
        while active_prompts and used < token_budget:
            p = active_prompts.pop(0)
            take = min(p, token_budget - used)
            used += take
            if take < p:
                active_prompts.insert(0, p - take)
        steps += 1
    return steps
