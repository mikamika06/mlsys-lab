def pack_step(decodes, prefills, token_budget):
    selected_decodes = list(decodes)
    budget_used = sum(1 for _ in selected_decodes)
    selected_prefills = []
    for p in prefills:
        remaining_tokens = p["remaining"]
        take = min(remaining_tokens, max(0, token_budget - budget_used))
        if take > 0:
            selected_prefills.append({"id": p["id"], "take": take})
            budget_used += take
        if budget_used >= token_budget:
            break
    return selected_decodes, selected_prefills
