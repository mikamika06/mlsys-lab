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
