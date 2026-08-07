def pack_step(decodes_count, prefill_list, budget):
    used = decodes_count
    allocated_prefills = []
    remaining_prefills = list(prefill_list)
    rem_budget = budget - used
    while rem_budget > 0 and remaining_prefills:
        p = remaining_prefills.pop(0)
        if p <= rem_budget:
            allocated_prefills.append((0, p))
            rem_budget -= p
        else:
            allocated_prefills.append((0, rem_budget))
            remaining_prefills.insert(0, p - rem_budget)
            rem_budget = 0
    return decodes_count, allocated_prefills, remaining_prefills
