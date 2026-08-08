def diagnose_nvtx_mismatches(events):
    stacks = {}
    ranges = []
    negative_ranges = []
    unclosed_pushes = []
    orphan_pops = []

    for evt in events:
        tid = evt["thread_id"]
        etype = evt["type"]
        ts = evt["timestamp"]

        if etype == "push":
            stacks.setdefault(tid, []).append(evt)
        elif etype == "pop":
            if tid in stacks and len(stacks[tid]) > 0:
                push_evt = stacks[tid].pop()
                dur = ts - push_evt["timestamp"]
                r = {
                    "name": push_evt["name"],
                    "thread_id": tid,
                    "start": push_evt["timestamp"],
                    "end": ts,
                    "duration": dur,
                }
                ranges.append(r)
                if dur < 0:
                    negative_ranges.append(r)
            else:
                orphan_pops.append({"thread_id": tid, "timestamp": ts})

    for tid, stack in stacks.items():
        for push_evt in stack:
            unclosed_pushes.append({
                "name": push_evt["name"],
                "thread_id": tid,
                "timestamp": push_evt["timestamp"],
            })

    return {
        "ranges": ranges,
        "negative_ranges": negative_ranges,
        "unclosed_pushes": unclosed_pushes,
        "orphan_pops": orphan_pops,
    }
