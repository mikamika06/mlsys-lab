def simulate_schedule(requests, capacity, policy="fcfs"):
    if policy == "fcfs":
        queue = list(requests)
        active_blocks = set()
        hits = 0
        total_tokens = 0
        waiting_times = []
        current_time = 0
        for req in queue:
            req_id, prompt_tokens, arrival_time = req["id"], req["tokens"], req["arrival"]
            wait = max(0, current_time - arrival_time)
            waiting_times.append(wait)
            matched = 0
            for t in prompt_tokens:
                if t in active_blocks:
                    matched += 1
                else:
                    active_blocks.add(t)
            hits += matched
            total_tokens += len(prompt_tokens)
            current_time = max(current_time, arrival_time) + 1
        hit_rate = hits / max(1, total_tokens)
        max_wait = max(waiting_times) if waiting_times else 0
        return {"hit_rate": hit_rate, "max_wait": max_wait}
    elif policy == "lpm":
        pending = list(requests)
        active_blocks = set()
        hits = 0
        total_tokens = 0
        waiting_times = []
        current_time = 0
        while pending:
            best_idx = 0
            best_match = -1
            for idx, req in enumerate(pending):
                match_count = sum(1 for t in req["tokens"] if t in active_blocks or len(active_blocks) < capacity)
                if match_count > best_match:
                    best_match = match_count
                    best_idx = idx
            req = pending.pop(best_idx)
            wait = max(0, current_time - req["arrival"])
            waiting_times.append(wait)
            matched = 0
            for t in req["tokens"]:
                if t in active_blocks:
                    matched += 1
                else:
                    if len(active_blocks) >= capacity:
                        active_blocks.pop()
                    active_blocks.add(t)
            hits += matched
            total_tokens += len(req["tokens"])
            current_time = max(current_time, req["arrival"]) + 1
        hit_rate = hits / max(1, total_tokens)
        max_wait = max(waiting_times) if waiting_times else 0
        return {"hit_rate": hit_rate, "max_wait": max_wait}
    raise ValueError(f"Unknown policy {policy}")
