from routing.penalty import compute_overlap, compute_staleness


def routing_cost(overlap_len, total_len, staleness, transfer_cost_per_token, compute_cost_per_token):
    missing_len = max(0, total_len - overlap_len)
    base_cost = float(overlap_len) * float(transfer_cost_per_token) + float(missing_len) * float(compute_cost_per_token)
    penalty = base_cost * float(staleness)
    return base_cost + penalty


def select_best_worker(request_tokens, workers, transfer_cost_per_token, compute_cost_per_token, current_tick):
    best_wid = None
    min_cost = float("inf")
    for w in workers:
        wid = w["worker_id"]
        tokens = w["cached_tokens"]
        last_tick = w["last_access_tick"]
        decay = w.get("decay_factor", 0.01)
        overlap = compute_overlap(request_tokens, tokens)
        staleness = compute_staleness(last_tick, current_tick, decay)
        cost = routing_cost(overlap, len(request_tokens), staleness, transfer_cost_per_token, compute_cost_per_token)
        if cost < min_cost:
            min_cost = cost
            best_wid = wid
    return best_wid
