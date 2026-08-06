def simulate_queue(requests, max_slots, queue_capacity):
    active_slots = 0
    queue = []
    completed = []
    rejected = []

    events = []
    for req in requests:
        events.append((req["arrival"], "arrive", req))
        events.append((req["arrival"] + req["duration"], "depart", req))

    events.sort(key=lambda x: (x[0], 0 if x[1] == "depart" else 1))

    current_time = 0.0
    for time_val, ev_type, req in events:
        current_time = time_val
        if ev_type == "depart":
            if req.get("in_slot", False):
                active_slots -= 1
                completed.append(req)
                if queue:
                    next_req = queue.pop(0)
                    next_req["in_slot"] = True
                    active_slots += 1
        elif ev_type == "arrive":
            if active_slots < max_slots:
                active_slots += 1
                req["in_slot"] = True
            elif len(queue) < queue_capacity:
                req["in_slot"] = False
                queue.append(req)
            else:
                req["in_slot"] = False
                rejected.append(req)

    return {
        "completed": [r["id"] for r in completed],
        "rejected": [r["id"] for r in rejected]
    }
