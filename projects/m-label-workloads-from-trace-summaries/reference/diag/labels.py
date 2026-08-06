def classify_op(event):
    if event["flops"] == 0 and event["bytes"] == 0:
        return "overhead"
    if event["launch_delay_us"] > event["dur_us"]:
        return "launch_bound"
    ai = event["flops"] / max(event["bytes"], 1)
    if ai > 100.0:
        return "compute_bound"
    return "memory_bound"


def label_workloads(trace):
    return [classify_op(ev) for ev in trace]
