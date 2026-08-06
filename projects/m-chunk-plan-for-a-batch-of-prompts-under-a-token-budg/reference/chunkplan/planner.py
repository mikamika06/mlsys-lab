def plan_chunks(prompts, token_budget):
    sorted_prompts = sorted(enumerate(prompts), key=lambda x: x[1])
    allocations = [0] * len(prompts)
    remaining_budget = token_budget
    
    active = []
    for idx, length in sorted_prompts:
        active.append((idx, length, 0))
        
    while remaining_budget > 0 and active:
        next_active = []
        fair_share = max(1, remaining_budget // len(active))
        used_in_round = 0
        
        for idx, total_len, current_len in active:
            needed = total_len - current_len
            if needed <= 0:
                continue
            alloc = min(needed, fair_share, remaining_budget)
            if alloc > 0:
                allocations[idx] += alloc
                current_len += alloc
                remaining_budget -= alloc
                used_in_round += alloc
            if current_len < total_len:
                next_active.append((idx, total_len, current_len))
        
        active = next_active
        if used_in_round == 0:
            break
            
    return allocations
