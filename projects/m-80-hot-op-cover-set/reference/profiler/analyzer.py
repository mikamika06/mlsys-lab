def detect_warmup(events):
    runs = [e for e in events if e.get("name") == "model_run"]
    if not runs:
        return 0
    durs = sorted([r["dur"] for r in runs])
    median_dur = durs[len(durs) // 2]
    threshold = 1.2 * median_dur
    for r in runs:
        if r["dur"] <= threshold:
            return r["ts"]
    return 0


def hot_op_cover(events, threshold=0.8):
    ts = detect_warmup(events)
    steady = [e for e in events if e.get("ts", 0) >= ts]
    op_durs = {}
    for e in steady:
        if e.get("cat") == "Node" and "args" in e and "op_name" in e["args"]:
            op = e["args"]["op_name"]
            op_durs[op] = op_durs.get(op, 0) + e["dur"]
    
    total = sum(op_durs.values())
    target = threshold * total
    sorted_ops = sorted(op_durs.items(), key=lambda x: (-x[1], x[0]))
    
    cover = set()
    accum = 0
    for op, dur in sorted_ops:
        cover.add(op)
        accum += dur
        if accum >= target:
            break
    return cover


def attribute_speedup(events_before, events_after):
    def avg_per_run(events):
        ts = detect_warmup(events)
        steady = [e for e in events if e.get("ts", 0) >= ts]
        runs = [e for e in steady if e.get("name") == "model_run"]
        n_runs = len(runs) if runs else 1
        op_durs = {}
        for e in steady:
            if e.get("cat") == "Node" and "args" in e and "op_name" in e["args"]:
                op = e["args"]["op_name"]
                op_durs[op] = op_durs.get(op, 0) + e["dur"]
        return {op: dur / n_runs for op, dur in op_durs.items()}
        
    before_avg = avg_per_run(events_before)
    after_avg = avg_per_run(events_after)
    
    all_ops = set(before_avg.keys()) | set(after_avg.keys())
    return {op: before_avg.get(op, 0.0) - after_avg.get(op, 0.0) for op in all_ops}
