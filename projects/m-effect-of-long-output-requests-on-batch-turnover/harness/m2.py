import ref

def check(workdir):
    from turnover.schedule import schedule_batch
    active = [{"id": 10, "remaining_tokens": 1000}]
    pending = [{"id": 11, "remaining_tokens": 10}, {"id": 12, "remaining_tokens": 10}]

    res_fcfs = schedule_batch(active, pending, capacity=2, policy="fcfs")
    res_budget = schedule_batch(active, pending, capacity=2, policy="preemptive_budget")

    turnover_improved = 1.0 if len(res_budget) == 2 and any(r["id"] in (11, 12) for r in res_budget) else 0.0
    latency_bounded = 1.0 if res_budget[0]["remaining_tokens"] < 1000 else 0.0

    return {
        "turnover_improved": turnover_improved,
        "latency_bounded": latency_bounded
    }
