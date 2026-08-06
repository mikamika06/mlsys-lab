import random


def build_bucket_plan(params, bucket_cap_bytes):
    buckets = []
    current_bucket = []
    current_bytes = 0
    for p in reversed(params):
        p_bytes = p["numel"] * p["element_size"]
        if current_bucket and (current_bytes + p_bytes > bucket_cap_bytes):
            buckets.append(current_bucket)
            current_bucket = [p["name"]]
            current_bytes = p_bytes
        else:
            current_bucket.append(p["name"])
            current_bytes += p_bytes
    if current_bucket:
        buckets.append(current_bucket)
    return buckets


def _merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_overlap_fraction(events):
    total_comm_time = 0.0
    total_overlap_time = 0.0
    ranks = set(e["rank"] for e in events)
    for r in sorted(ranks):
        rank_events = [e for e in events if e["rank"] == r]
        comm_raw = [(e["start"], e["end"]) for e in rank_events if e["type"] == "comm"]
        compute_raw = [(e["start"], e["end"]) for e in rank_events if e["type"] == "compute"]
        comm_merged = _merge_intervals(comm_raw)
        compute_merged = _merge_intervals(compute_raw)
        for c_s, c_e in comm_merged:
            total_comm_time += (c_e - c_s)
            for p_s, p_e in compute_merged:
                inter_s = max(c_s, p_s)
                inter_e = min(c_e, p_e)
                if inter_e > inter_s:
                    total_overlap_time += (inter_e - inter_s)
    if total_comm_time == 0.0:
        return 0.0
    return total_overlap_time / total_comm_time


def check_rank_plan_consistency(rank_plans):
    if not rank_plans:
        return {"consistent": True, "mismatched_ranks": []}
    ranks = sorted(rank_plans.keys())
    ref_rank = ranks[0]
    ref_plan = rank_plans[ref_rank]
    mismatched = []
    for r in ranks[1:]:
        if rank_plans[r] != ref_plan:
            mismatched.append(r)
    return {
        "consistent": len(mismatched) == 0,
        "mismatched_ranks": mismatched
    }


def generate_param_configs(seed=42):
    rng = random.Random(seed)
    configs = []
    for i in range(5):
        num_params = rng.randint(5, 15)
        params = []
        for j in range(num_params):
            params.append({
                "name": f"layer_{j}.weight",
                "numel": rng.randint(100, 1000),
                "element_size": rng.choice([2, 4])
            })
        cap = rng.randint(1000, 4000)
        configs.append((params, cap))
    return configs


def generate_trace_events(seed=123):
    rng = random.Random(seed)
    events = []
    for rank in range(4):
        t = 0.0
        for _ in range(10):
            comp_dur = rng.uniform(10.0, 50.0)
            events.append({"rank": rank, "type": "compute", "start": t, "end": t + comp_dur})
            if rng.random() > 0.3:
                comm_offset = rng.uniform(0.0, comp_dur / 2.0)
                comm_dur = rng.uniform(5.0, 30.0)
                comm_start = t + comm_offset
                events.append({"rank": rank, "type": "comm", "start": comm_start, "end": comm_start + comm_dur})
            t += comp_dur + rng.uniform(2.0, 10.0)
    return events
