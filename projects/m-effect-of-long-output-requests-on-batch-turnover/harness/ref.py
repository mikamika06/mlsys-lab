def get_test_requests():
    return [
        {"id": 1, "initial_tokens": 10, "remaining_tokens": 10},
        {"id": 2, "initial_tokens": 500, "remaining_tokens": 500},
        {"id": 3, "initial_tokens": 15, "remaining_tokens": 15},
        {"id": 4, "initial_tokens": 800, "remaining_tokens": 800},
        {"id": 5, "initial_tokens": 20, "remaining_tokens": 20},
    ]

def compute_turnover_metrics(requests, batch_capacity, steps):
    active = []
    pending = list(requests)
    completed = []
    turnover_history = []

    for step in range(steps):
        while len(active) < batch_capacity and pending:
            active.append(pending.pop(0))

        still_active = []
        completed_this_step = 0
        for req in active:
            req["remaining_tokens"] -= 1
            if req["remaining_tokens"] <= 0:
                completed.append(req)
                completed_this_step += 1
            else:
                still_active.append(req)
        active = still_active
        turnover_history.append(completed_this_step)

    avg_turnover = sum(turnover_history) / len(turnover_history) if turnover_history else 0.0
    return {
        "completed_count": len(completed),
        "avg_turnover": avg_turnover,
        "max_active_span": max((r["initial_tokens"] for r in completed), default=0)
    }

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
