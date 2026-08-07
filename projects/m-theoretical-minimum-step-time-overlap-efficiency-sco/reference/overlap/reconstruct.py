"""Trace event timeline reconstruction."""


def reconstruct_timeline(events):
    if not events:
        return {
            "total_span": 0.0,
            "compute_only": 0.0,
            "comm_only": 0.0,
            "overlapped": 0.0,
            "idle": 0.0,
        }

    points = set()
    for e in events:
        points.add(e["start"])
        points.add(e["end"])
    sorted_pts = sorted(points)

    total_span = max(e["end"] for e in events) - min(e["start"] for e in events)
    compute_only = 0.0
    comm_only = 0.0
    overlapped = 0.0
    idle = 0.0

    min_t = min(e["start"] for e in events)
    max_t = max(e["end"] for e in events)

    for i in range(len(sorted_pts) - 1):
        t0, t1 = sorted_pts[i], sorted_pts[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue

        has_compute = any(
            e["kind"] == "compute" and e["start"] <= t0 and e["end"] >= t1
            for e in events
        )
        has_comm = any(
            e["kind"] == "comm" and e["start"] <= t0 and e["end"] >= t1
            for e in events
        )

        if has_compute and has_comm:
            overlapped += dt
        elif has_compute:
            compute_only += dt
        elif has_comm:
            comm_only += dt
        else:
            if t0 >= min_t and t1 <= max_t:
                idle += dt

    return {
        "total_span": float(total_span),
        "compute_only": float(compute_only),
        "comm_only": float(comm_only),
        "overlapped": float(overlapped),
        "idle": float(idle),
    }
