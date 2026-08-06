def schedule_batch(active_batch, pending_requests, capacity, policy="fcfs"):
    if policy == "fcfs":
        combined = active_batch + pending_requests
        return combined[:capacity]
    elif policy == "preemptive_budget":
        combined = active_batch + pending_requests
        combined.sort(key=lambda r: r["remaining_tokens"])
        return combined[:capacity]
    else:
        raise ValueError(f"Unknown policy {policy}")
