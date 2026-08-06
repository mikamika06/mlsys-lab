def simulate_schedule(requests, max_batch_size):
    active = []
    queue = list(requests)
    timeline = []
    step = 0
    while queue or active:
        while queue and len(active) < max_batch_size:
            req = queue.pop(0)
            active.append({"id": req["id"], "remaining": req["output_len"]})
        step += 1
        completed = []
        for req in active:
            req["remaining"] -= 1
            if req["remaining"] == 0:
                completed.append(req["id"])
        timeline.append({"step": step, "completed": sorted(completed), "active_count": len(active)})
        active = [req for req in active if req["remaining"] > 0]
    return timeline
