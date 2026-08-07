def parse_trace_events(raw_events):
    x_events = []
    stacks = {}
    unmatched = 0
    for ev in raw_events:
        ph = ev.get("ph")
        if ph == "X":
            x_events.append(ev)
        elif ph == "B":
            key = (ev.get("pid"), ev.get("tid"))
            stacks.setdefault(key, []).append(ev)
        elif ph == "E":
            key = (ev.get("pid"), ev.get("tid"))
            st = stacks.get(key, [])
            if st:
                st.pop()
            else:
                unmatched += 1
    for st in stacks.values():
        unmatched += len(st)
    return {
        "x_events": x_events,
        "is_truncated": unmatched > 0,
        "unmatched_b_count": unmatched
    }
