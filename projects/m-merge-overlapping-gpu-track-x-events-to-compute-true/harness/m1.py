import ref

def check(workdir):
    from gpuprof.events import parse_trace_events

    out = {"traces_parsed": 0.0, "truncation_detected": 0.0}
    total_traces = len(ref.TRACES)
    parsed_ok = 0
    trunc_ok = 0

    for i, trace in enumerate(ref.TRACES):
        want = ref.ref_parse_trace_events(trace)
        try:
            got = parse_trace_events(trace)
        except Exception as e:
            out["_note"] = f"trace {i} raised {type(e).__name__}: {str(e)[:100]}"
            return out

        if not isinstance(got, dict):
            out["_note"] = f"trace {i}: expected dict return, got {type(got).__name__}"
            return out

        x_events_match = len(got.get("x_events", [])) == len(want["x_events"])
        count_match = got.get("unmatched_b_count") == want["unmatched_b_count"]
        trunc_match = got.get("is_truncated") == want["is_truncated"]

        if x_events_match and count_match:
            parsed_ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got x_events={len(got.get('x_events', []))}, unmatched={got.get('unmatched_b_count')}; want x_events={len(want['x_events'])}, unmatched={want['unmatched_b_count']}"

        if trunc_match:
            trunc_ok += 1

    out["traces_parsed"] = 1.0 if parsed_ok == total_traces else 0.0
    out["truncation_detected"] = 1.0 if trunc_ok == total_traces else 0.0
    return out
