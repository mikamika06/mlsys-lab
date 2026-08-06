def analyze_preemption_logs(log_events, current_args):
    """Analyzes preemption logs and recommends parameter changes."""
    preempt_counts = {}
    finished_count = 0
    total_preemptions = 0
    for event in log_events:
        req_id = event.get("req_id")
        evt = event.get("event")
        if evt == "preempt":
            preempt_counts[req_id] = preempt_counts.get(req_id, 0) + 1
            total_preemptions += 1
        elif evt == "finish":
            finished_count += 1

    has_repeated_preempt = any(c >= 2 for c in preempt_counts.values())
    high_preempt_ratio = total_preemptions > max(1, finished_count) * 0.5 if log_events else False
    is_thrashing = has_repeated_preempt or high_preempt_ratio

    recommendation = {}
    if is_thrashing:
        gpu_util = current_args.get("gpu_memory_utilization", 0.90)
        if round(gpu_util + 0.05, 2) <= 0.95:
            recommendation["gpu_memory_utilization"] = round(gpu_util + 0.05, 2)
        else:
            max_seqs = current_args.get("max_num_seqs", 256)
            recommendation["max_num_seqs"] = max(1, max_seqs // 2)

    return {"is_thrashing": is_thrashing, "recommendation": recommendation}
