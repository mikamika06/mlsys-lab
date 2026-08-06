import random

def generate_fixtures():
    random.seed(42)
    trace_data = {}
    for rank in range(4):
        rank_key = f"rank_{rank}"
        events = []
        for i in range(10):
            base = i * 100.0
            events.append({"type": "compute", "start": base + 0.0, "end": base + 60.0})
            events.append({"type": "comm", "start": base + 40.0, "end": base + 90.0})
            duration = 15.0 if rank == 2 else 5.0
            events.append({"type": "allreduce", "start": base + 60.0, "end": base + 60.0 + duration})
        trace_data[rank_key] = events

    timings = {"step_time": 0.250}
    model_params = {
        "msg_size_bytes": 268435456,
        "world_size": 8,
        "bandwidth_bytes_per_sec": 26843545600
    }
    return trace_data, timings, model_params

def ref_compute_overlap(trace_data):
    results = {}
    for rank, events in trace_data.items():
        comp_raw = [(ev["start"], ev["end"]) for ev in events if ev.get("type") == "compute"]
        comm_raw = [(ev["start"], ev["end"]) for ev in events if ev.get("type") == "comm"]

        def merge(invs):
            if not invs:
                return []
            s_invs = sorted(invs, key=lambda x: x[0])
            m = [s_invs[0]]
            for cur in s_invs[1:]:
                if cur[0] <= m[-1][1]:
                    m[-1] = (m[-1][0], max(m[-1][1], cur[1]))
                else:
                    m.append(cur)
            return m

        comp_m = merge(comp_raw)
        comm_m = merge(comm_raw)

        comp_tot = sum(e - s for s, e in comp_m)
        if comp_tot <= 0.0:
            results[rank] = 0.0
            continue

        overlap = 0.0
        for cs, ce in comp_m:
            for ms, me in comm_m:
                overlap += max(0.0, min(ce, me) - max(cs, ms))

        results[rank] = (overlap / comp_tot) * 100.0
    return results

def ref_identify_straggler(trace_data):
    rank_totals = {}
    for rank, events in trace_data.items():
        total = sum(
            ev["end"] - ev["start"]
            for ev in events
            if ev.get("type") == "allreduce"
        )
        rank_totals[rank] = total
    if not rank_totals:
        return None
    return max(rank_totals, key=rank_totals.get)

def ref_compute_comm_bound_ratio(timings, model_params):
    step_time = float(timings["step_time"])
    msg_size = float(model_params["msg_size_bytes"])
    world_size = float(model_params["world_size"])
    bandwidth = float(model_params["bandwidth_bytes_per_sec"])
    comm_time = 2.0 * ((world_size - 1.0) / world_size) * (msg_size / bandwidth)
    return (comm_time / step_time) * 100.0
