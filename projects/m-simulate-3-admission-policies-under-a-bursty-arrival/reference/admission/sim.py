def simulate(trace, policy, max_len=0, max_wait=0):
    out = []
    free_at = 0
    active = []
    for req in trace:
        t = req["arrival"]
        active = [c for c in active if c > t]
        wait = max(0, free_at - t)

        admit = True
        if policy == "queue_limit" and len(active) >= max_len:
            admit = False
        elif policy == "time_limit" and wait > max_wait:
            admit = False

        if admit:
            free_at = t + wait + req["cost"]
            active.append(free_at)
            out.append({"id": req["id"], "admitted": True, "wait": wait, "completion": free_at})
        else:
            out.append({"id": req["id"], "admitted": False, "wait": 0, "completion": 0})
    return out

def find_trigger(trace, max_wait):
    free_at = 0
    for req in trace:
        t = req["arrival"]
        wait = max(0, free_at - t)
        if wait > max_wait:
            return req["id"]
        free_at = t + wait + req["cost"]
    return -1
