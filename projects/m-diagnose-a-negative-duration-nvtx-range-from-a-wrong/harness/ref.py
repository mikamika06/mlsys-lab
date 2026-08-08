NVTX_EVENTS = [
    {"id": 1, "thread_id": 100, "type": "push", "name": "data_loading", "timestamp": 1000},
    {"id": 2, "thread_id": 100, "type": "pop", "name": None, "timestamp": 1200},
    {"id": 3, "thread_id": 101, "type": "push", "name": "fused_attention", "timestamp": 1500},
    {"id": 4, "thread_id": 101, "type": "push", "name": "qkv_proj", "timestamp": 1520},
    {"id": 5, "thread_id": 101, "type": "pop", "name": None, "timestamp": 1510},
    {"id": 6, "thread_id": 102, "type": "pop", "name": None, "timestamp": 1600},
    {"id": 7, "thread_id": 101, "type": "pop", "name": None, "timestamp": 1800},
    {"id": 8, "thread_id": 100, "type": "push", "name": "backward_step", "timestamp": 2000},
]

MAC_TRACE_EVENTS = [
    {"name": "forward", "ph": "X", "ts": 0.0, "dur": 1000.0, "tid": 10},
    {"name": "attention", "ph": "X", "ts": 50.0, "dur": 400.0, "tid": 10},
    {"name": "layernorm", "ph": "X", "ts": 100.0, "dur": 50.0, "tid": 10},
    {"name": "mlp", "ph": "X", "ts": 500.0, "dur": 400.0, "tid": 10},
    {"name": "layernorm", "ph": "X", "ts": 520.0, "dur": 50.0, "tid": 10},
    {"name": "optimizer_step", "ph": "X", "ts": 1100.0, "dur": 300.0, "tid": 10},
]

TARGET_PHASES = ["forward", "attention", "mlp", "layernorm", "optimizer_step"]


def diagnose_nvtx_mismatches(events):
    stacks = {}
    ranges = []
    negative_ranges = []
    unclosed_pushes = []
    orphan_pops = []

    for evt in events:
        tid = evt["thread_id"]
        etype = evt["type"]
        ts = evt["timestamp"]

        if etype == "push":
            stacks.setdefault(tid, []).append(evt)
        elif etype == "pop":
            if tid in stacks and len(stacks[tid]) > 0:
                push_evt = stacks[tid].pop()
                dur = ts - push_evt["timestamp"]
                r = {
                    "name": push_evt["name"],
                    "thread_id": tid,
                    "start": push_evt["timestamp"],
                    "end": ts,
                    "duration": dur,
                }
                ranges.append(r)
                if dur < 0:
                    negative_ranges.append(r)
            else:
                orphan_pops.append({"thread_id": tid, "timestamp": ts})

    for tid, stack in stacks.items():
        for push_evt in stack:
            unclosed_pushes.append({
                "name": push_evt["name"],
                "thread_id": tid,
                "timestamp": push_evt["timestamp"],
            })

    return {
        "ranges": ranges,
        "negative_ranges": negative_ranges,
        "unclosed_pushes": unclosed_pushes,
        "orphan_pops": orphan_pops,
    }


def analyze_mac_trace(trace_events, target_phases):
    phase_totals = {p: 0.0 for p in target_phases}
    phase_selves = {p: 0.0 for p in target_phases}

    def contains(outer, inner):
        o_start, o_end = outer["ts"], outer["ts"] + outer["dur"]
        i_start, i_end = inner["ts"], inner["ts"] + inner["dur"]
        return o_start <= i_start and i_end <= o_end

    for evt in trace_events:
        name = evt.get("name")
        if name not in target_phases:
            continue

        dur = float(evt.get("dur", 0.0))
        tid = evt.get("tid")
        phase_totals[name] += dur

        descendants = []
        for other in trace_events:
            if other is evt or other.get("tid") != tid:
                continue
            if contains(evt, other):
                descendants.append(other)

        immediate_children = []
        for d in descendants:
            is_nested = False
            for parent_candidate in descendants:
                if parent_candidate is d:
                    continue
                if contains(parent_candidate, d):
                    p_dur = parent_candidate.get("dur", 0.0)
                    d_dur = d.get("dur", 0.0)
                    if p_dur > d_dur or (
                        p_dur == d_dur
                        and trace_events.index(parent_candidate)
                        < trace_events.index(d)
                    ):
                        is_nested = True
                        break
            if not is_nested:
                immediate_children.append(d)

        child_dur_sum = sum(
            float(c.get("dur", 0.0)) for c in immediate_children
        )
        self_time = max(0.0, dur - child_dur_sum)
        phase_selves[name] += self_time

    rankings = sorted(target_phases, key=lambda p: (-phase_selves[p], p))

    phase_metrics = {}
    for rank, p in enumerate(rankings, 1):
        phase_metrics[p] = {
            "total_time": phase_totals[p],
            "self_time": phase_selves[p],
            "rank": rank,
        }

    return {"rankings": rankings, "phase_metrics": phase_metrics}
