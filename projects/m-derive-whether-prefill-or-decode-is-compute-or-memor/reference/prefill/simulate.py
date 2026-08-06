def simulate_batch(cfg):
    """Simulate chunked-prefill batch composition."""
    budget = cfg["token_budget"]
    decodes = cfg["decodes"]
    prefills = cfg["prefills"]
    dec_tokens_used = len(decodes)
    remaining_budget = budget - dec_tokens_used
    if remaining_budget < 0:
        selected_decodes = decodes[:budget]
        selected_prefills = []
        return {"decodes": selected_decodes, "prefills": selected_prefills, "tokens_used": budget}
    selected_decodes = list(decodes)
    selected_prefills = []
    for p in prefills:
        if remaining_budget <= 0:
            break
        alloc = min(p["remaining_tokens"], p["max_chunk"], remaining_budget)
        if alloc > 0:
            selected_prefills.append({"id": p["id"], "allocated_tokens": alloc})
            remaining_budget -= alloc
    tokens_used = budget - remaining_budget
    return {"decodes": selected_decodes, "prefills": selected_prefills, "tokens_used": tokens_used}
