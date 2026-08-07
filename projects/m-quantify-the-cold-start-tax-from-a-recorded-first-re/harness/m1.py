import ref


def check(workdir):
    from coldstart.trace import parse_trace

    trace_data = ref.generate_trace()
    want = ref.parse_trace(trace_data)
    got = parse_trace(trace_data)

    out = {"sorting_match": 0.0}
    if got == want:
        out["sorting_match"] = 1.0
    else:
        out["_note"] = "parse_trace sorting mismatch"
    return out
