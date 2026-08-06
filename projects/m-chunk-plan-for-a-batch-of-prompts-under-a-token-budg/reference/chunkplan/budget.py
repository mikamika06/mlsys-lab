def plan_chunks(prompt_lengths, token_budget):
    remaining = list(prompt_lengths)
    plan = []
    tb = token_budget
    while remaining and tb > 0:
        take = min(remaining[0], tb)
        plan.append(take)
        tb -= take
        if take == remaining[0]:
            remaining.pop(0)
    return plan
