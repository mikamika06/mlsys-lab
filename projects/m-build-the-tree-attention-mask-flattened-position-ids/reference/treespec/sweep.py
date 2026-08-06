def reconstruct_sweep_stats(records):
    steps_map = {}
    for rec in records:
        step = rec["step"]
        width = rec["width"]
        accepted = rec["accepted_count"]
        steps_map.setdefault(step, []).append((width, accepted))
    
    result = []
    for step in sorted(steps_map.keys()):
        vals = steps_map[step]
        avg_accepted = sum(a for _, a in vals) / len(vals)
        avg_width = sum(w for w, _ in vals) / len(vals)
        result.append({"step": step, "avg_accepted": avg_accepted, "avg_width": avg_width})
    return result
