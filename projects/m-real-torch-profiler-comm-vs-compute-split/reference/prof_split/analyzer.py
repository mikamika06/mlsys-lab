def extract_intervals(trace_data):
    compute = []
    comm = []
    for ev in trace_data.get("traceEvents", []):
        name = ev.get("name", "")
        dur = ev.get("dur", 0)
        ts = ev.get("ts", 0)
        if "nccl" in name.lower() or "comm" in name.lower():
            comm.append((ts, ts + dur))
        elif "aten" in name.lower() or "matmul" in name.lower() or "kernel" in name.lower():
            compute.append((ts, ts + dur))
    return sorted(compute), sorted(comm)


def compute_overlap_percentage(compute_intervals, comm_intervals):
    if not compute_intervals or not comm_intervals:
        return 0.0

    merged = []
    for start, end in sorted(compute_intervals):
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    comp_union = sum(e - s for s, e in merged)
    if comp_union == 0:
        return 0.0

    overlap_total = 0
    for cs, ce in compute_intervals:
        for oms, ome in comm_intervals:
            o_start = max(cs, oms)
            o_end = min(ce, ome)
            if o_start < o_end:
                overlap_total += (o_end - o_start)

    ratio = float(overlap_total) / float(comp_union)
    return min(100.0, max(0.0, ratio * 100.0))
