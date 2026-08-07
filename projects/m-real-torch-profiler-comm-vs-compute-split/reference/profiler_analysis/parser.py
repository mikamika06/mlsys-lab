def parse_comm_compute_split(trace):
    compute_time = 0.0
    comm_time = 0.0
    for ev in trace.get("traceEvents", []):
        cat = ev.get("cat")
        dur = ev.get("dur", 0.0)
        if cat == "compute":
            compute_time += dur
        elif cat == "comm":
            comm_time += dur
    return {"compute_time": compute_time, "comm_time": comm_time}
