def select_victim(active_requests, policy):
    if not active_requests:
        return None
    if policy == "lru":
        return min(active_requests, key=lambda r: r["last_active_time"])["id"]
    elif policy == "longest_prefill":
        return max(active_requests, key=lambda r: r["prefill_len"])["id"]
    elif policy == "fifo":
        return min(active_requests, key=lambda r: r["arrival_time"])["id"]
    else:
        return active_requests[0]["id"]
