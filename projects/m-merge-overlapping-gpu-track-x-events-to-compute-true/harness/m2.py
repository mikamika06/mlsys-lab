import ref

def check(workdir):
    from gpuprof.busy import compute_gpu_busy_time
    from gpuprof.events import parse_trace_events
    from gpuprof.latency import compute_host_to_device_latencies

    out = {"busy_time_rel_err": 1.0, "latencies_matched": 0.0}
    max_rel_err = 0.0
    latency_ok = 0
    total_traces = len(ref.TRACES)

    for i, trace in enumerate(ref.TRACES):
        want_parsed = ref.ref_parse_trace_events(trace)
        try:
            got_parsed = parse_trace_events(trace)
        except Exception as e:
            out["_note"] = f"parse_trace_events trace {i} raised {type(e).__name__}"
            return out

        x_events = got_parsed.get("x_events", [])

        want_busy = ref.ref_compute_gpu_busy_time(want_parsed["x_events"])
        try:
            got_busy = compute_gpu_busy_time(x_events)
        except Exception as e:
            out["_note"] = f"compute_gpu_busy_time trace {i} raised {type(e).__name__}"
            return out

        err = abs(got_busy - want_busy) / max(1.0, abs(want_busy))
        max_rel_err = max(max_rel_err, err)

        if i == 2:
            want_busy_st1 = ref.ref_compute_gpu_busy_time(want_parsed["x_events"], stream_ids={1})
            try:
                got_busy_st1 = compute_gpu_busy_time(x_events, stream_ids={1})
            except Exception as e:
                out["_note"] = f"compute_gpu_busy_time filtered trace {i} raised {type(e).__name__}"
                return out
            err_st1 = abs(got_busy_st1 - want_busy_st1) / max(1.0, abs(want_busy_st1))
            max_rel_err = max(max_rel_err, err_st1)

        want_lat = ref.ref_compute_host_to_device_latencies(want_parsed["x_events"])
        try:
            got_lat = compute_host_to_device_latencies(x_events)
        except Exception as e:
            out["_note"] = f"compute_host_to_device_latencies trace {i} raised {type(e).__name__}"
            return out

        if isinstance(got_lat, dict) and len(got_lat) == len(want_lat):
            match = True
            for k, v in want_lat.items():
                if k not in got_lat or abs(got_lat[k] - v) > 1e-5:
                    match = False
                    break
            if match:
                latency_ok += 1
            elif "_note" not in out:
                out["_note"] = f"trace {i}: got latencies {got_lat}, want {want_lat}"
        elif "_note" not in out:
            out["_note"] = f"trace {i}: latencies len mismatch got {len(got_lat) if isinstance(got_lat, dict) else type(got_lat)}, want {len(want_lat)}"

    out["busy_time_rel_err"] = float(max_rel_err)
    out["latencies_matched"] = 1.0 if latency_ok == total_traces else 0.0
    return out
