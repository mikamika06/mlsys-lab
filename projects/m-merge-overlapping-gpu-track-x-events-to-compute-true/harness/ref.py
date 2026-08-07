def _make_traces():
    t0 = [
        {"ph": "X", "cat": "host_op", "name": "launch_kernel_1", "ts": 100.0, "dur": 10.0, "pid": 1, "tid": 1, "args": {"correlation_id": 101}},
        {"ph": "X", "cat": "gpu_op", "name": "kernel_1", "ts": 120.0, "dur": 50.0, "pid": 2, "tid": 1, "args": {"correlation_id": 101, "stream": 7}},
        {"ph": "X", "cat": "gpu_op", "name": "kernel_2", "ts": 140.0, "dur": 60.0, "pid": 2, "tid": 2, "args": {"correlation_id": 102, "stream": 8}},
        {"ph": "X", "cat": "host_op", "name": "launch_kernel_2", "ts": 110.0, "dur": 15.0, "pid": 1, "tid": 1, "args": {"correlation_id": 102}},
        {"ph": "B", "cat": "cpu_scope", "name": "step_0", "ts": 90.0, "pid": 1, "tid": 1},
        {"ph": "E", "cat": "cpu_scope", "name": "step_0", "ts": 210.0, "pid": 1, "tid": 1},
    ]

    t1 = [
        {"ph": "B", "cat": "cpu_scope", "name": "outer_scope", "ts": 10.0, "pid": 1, "tid": 1},
        {"ph": "B", "cat": "cpu_scope", "name": "inner_scope_1", "ts": 20.0, "pid": 1, "tid": 1},
        {"ph": "E", "cat": "cpu_scope", "name": "inner_scope_1", "ts": 30.0, "pid": 1, "tid": 1},
        {"ph": "B", "cat": "cpu_scope", "name": "inner_scope_2", "ts": 40.0, "pid": 1, "tid": 1},
        {"ph": "E", "cat": "cpu_scope", "name": "extra_unmatched_e", "ts": 50.0, "pid": 1, "tid": 2},
        {"ph": "X", "cat": "gpu_op", "name": "kernel_3", "ts": 100.0, "dur": 30.0, "pid": 2, "tid": 1, "args": {"stream": 7}},
    ]

    t2 = []
    for i in range(5):
        t2.append({"ph": "X", "cat": "host_op", "name": f"launch_{i}", "ts": 1000.0 + i * 20.0, "dur": 5.0, "pid": 10, "tid": 1, "args": {"correlation_id": 500 + i}})

    gpu_intervals = [
        (1050.0, 100.0, 1),
        (1080.0, 100.0, 2),
        (1200.0, 50.0, 1),
        (1220.0, 60.0, 2),
        (1300.0, 40.0, 1),
    ]
    for i, (ts, dur, st) in enumerate(gpu_intervals):
        t2.append({"ph": "X", "cat": "gpu_op", "name": f"gpu_kernel_{i}", "ts": ts, "dur": dur, "pid": 20, "tid": st, "args": {"correlation_id": 500 + i, "stream": st}})

    return [t0, t1, t2]

TRACES = _make_traces()

def ref_parse_trace_events(raw_events):
    x_events = []
    stacks = {}
    unmatched = 0
    for ev in raw_events:
        ph = ev.get("ph")
        if ph == "X":
            x_events.append(ev)
        elif ph == "B":
            key = (ev.get("pid"), ev.get("tid"))
            stacks.setdefault(key, []).append(ev)
        elif ph == "E":
            key = (ev.get("pid"), ev.get("tid"))
            st = stacks.get(key, [])
            if st:
                st.pop()
            else:
                unmatched += 1
    for st in stacks.values():
        unmatched += len(st)
    return {
        "x_events": x_events,
        "is_truncated": unmatched > 0,
        "unmatched_b_count": unmatched
    }

def ref_compute_gpu_busy_time(x_events, stream_ids=None):
    intervals = []
    for ev in x_events:
        if ev.get("cat") != "gpu_op":
            continue
        if stream_ids is not None:
            st = ev.get("args", {}).get("stream")
            if st not in stream_ids:
                continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur > 0:
            intervals.append((ts, ts + dur))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: x[0])
    merged = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append(cur_end - cur_start)
            cur_start, cur_end = start, end
    merged.append(cur_end - cur_start)
    return float(sum(merged))

def ref_compute_host_to_device_latencies(x_events):
    host_map = {}
    device_map = {}
    for ev in x_events:
        cid = ev.get("args", {}).get("correlation_id")
        if cid is None:
            continue
        cat = ev.get("cat")
        if cat == "host_op":
            host_map[cid] = float(ev.get("ts", 0.0))
        elif cat == "gpu_op":
            device_map[cid] = float(ev.get("ts", 0.0))
    res = {}
    for cid in host_map:
        if cid in device_map:
            res[cid] = device_map[cid] - host_map[cid]
    return res
