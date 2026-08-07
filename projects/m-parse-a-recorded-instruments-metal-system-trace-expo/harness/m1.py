import ref


def check(workdir):
    from mpslab.trace import parse_trace

    text = ref.generate_trace_text()
    want = ref.parse_trace(text)
    try:
        got = parse_trace(text)
    except Exception as e:
        return {"trace_parsed": 0.0, "_note": f"parse_trace raised {type(e).__name__}: {e}"}

    if got == want:
        return {"trace_parsed": 1.0}
    return {"trace_parsed": 0.0, "_note": f"got {got}, expected {want}"}
