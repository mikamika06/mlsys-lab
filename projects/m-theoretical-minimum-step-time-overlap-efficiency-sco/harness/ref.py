import numpy as np

SEEDS = [42, 101, 2026]


def generate_trace_events(seed=42):
    rng = np.random.default_rng(seed)
    num_events = rng.integers(15, 30)
    events = []
    t = 0.0
    for i in range(num_events):
        kind = rng.choice(["compute", "comm"])
        dur = float(rng.uniform(5.0, 50.0))
        stream = int(rng.choice([0, 1]))
        t += float(rng.uniform(0.0, 10.0))
        start = t
        end = start + dur
        events.append(
            {
                "id": i,
                "kind": kind,
                "start": round(start, 2),
                "end": round(end, 2),
                "stream": stream,
            }
        )
    return events


TRACES = [generate_trace_events(s) for s in SEEDS]


def generate_bucket_profiles(seed=42):
    rng = np.random.default_rng(seed)
    bucket_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    profiles = []
    for b in bucket_sizes:
        base_comm = 100.0 / (b**0.5) + 5.0 * b**0.3
        compute_time = 200.0
        overlap_factor = min(0.9, 0.1 * b**0.6)
        total_time = compute_time + base_comm * (1.0 - overlap_factor)
        profiles.append(
            {
                "bucket_size_mb": b,
                "total_step_time": round(float(total_time), 3),
                "comm_time": round(float(base_comm), 3),
                "compute_time": round(float(compute_time), 3),
            }
        )
    return profiles


BUCKET_DATASETS = [generate_bucket_profiles(s) for s in SEEDS]


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


def compute_metrics(events):
    recon = reconstruct_timeline(events)
    total_compute = recon["compute_only"] + recon["overlapped"]
    total_comm = recon["comm_only"] + recon["overlapped"]

    theoretical_min_step_time = max(total_compute, total_comm)
    actual_step_time = recon["total_span"]

    if total_comm > 0:
        overlap_efficiency_score = recon["overlapped"] / total_comm
    else:
        overlap_efficiency_score = 1.0

    return {
        "theoretical_min_step_time": float(theoretical_min_step_time),
        "actual_step_time": float(actual_step_time),
        "overlap_efficiency_score": float(overlap_efficiency_score),
    }


def find_saturation_point(bucket_profiles):
    best_b = None
    best_val = float("inf")
    for p in bucket_profiles:
        val = p["total_step_time"]
        if val < best_val:
            best_val = val
            best_b = p["bucket_size_mb"]
    return {"saturation_bucket_mb": best_b, "min_step_time": best_val}
