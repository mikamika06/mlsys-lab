def diagnose_negative_nvtx_range(events):
    stacks = {}
    for ev in events:
        tid = ev["thread_id"]
        if tid not in stacks:
            stacks[tid] = []
        if ev["type"] == "push":
            stacks[tid].append(ev)
        elif ev["type"] == "pop":
            if stacks[tid]:
                push_ev = stacks[tid].pop()
                dur = ev["timestamp"] - push_ev["timestamp"]
                if dur < 0:
                    return {
                        "push_id": push_ev["id"],
                        "pop_id": ev["id"],
                        "thread_id": tid,
                        "duration": dur,
                        "text": push_ev["text"],
                    }
    return None
