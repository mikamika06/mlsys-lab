def admit_requests(requests, budget):
    state = [
        {
            "id": r["id"],
            "kv": r["kv"],
            "tokens": r["tokens"],
        }
        for r in requests
    ]
    result = []
    while any(r["tokens"] > 0 for r in state):
        used = 0
        running = []
        for r in state:
            if r["tokens"] > 0 and used + r["kv"] <= budget:
                running.append(r)
                used += r["kv"]
        step = []
        for r in running:
            step.append(r["id"])
            r["tokens"] -= 1
        result.append(step)
    return result
