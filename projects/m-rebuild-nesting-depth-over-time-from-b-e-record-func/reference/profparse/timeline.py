def compute_nesting_depth(events, timestamps):
    depths = []
    for t in timestamps:
        active = 0
        for ev in events:
            if ev.get("ph") == "X":
                ts = ev.get("ts", 0)
                dur = ev.get("dur", 0)
                if ts <= t <= ts + dur:
                    active += 1
        depths.append(active)
    return depths
