def simulate_schedule(prompts, max_batched_tokens):
    queue = list(prompts)
    active_prefills = []
    decodes = []
    
    timeline = []
    time_step = 0
    
    while queue or active_prefills or decodes:
        while queue and len(active_prefills) + len(decodes) < 16:
            p = queue.pop(0)
            active_prefills.append({"id": p["id"], "total": p["length"], "processed": 0})
            
        tokens_used = 0
        current_step_decodes = len(decodes)
        tokens_used += current_step_decodes
        
        scheduled_prefills = []
        for p in active_prefills:
            rem = p["total"] - p["processed"]
            allowance = max_batched_tokens - tokens_used
            if allowance <= 0:
                break
            chunk = min(rem, allowance)
            if chunk > 0:
                tokens_used += chunk
                scheduled_prefills.append((p, chunk))
                
        timeline.append({
            "time": time_step,
            "decodes": current_step_decodes,
            "prefills": [(p["id"], c) for p, c in scheduled_prefills]
        })
        
        for p, c in scheduled_prefills:
            p["processed"] += c
            if p["processed"] >= p["total"]:
                decodes.append({"id": p["id"], "tokens": p["processed"]})
                
        active_prefills = [p for p in active_prefills if p["processed"] < p["total"]]
        time_step += 1
        
    return timeline
