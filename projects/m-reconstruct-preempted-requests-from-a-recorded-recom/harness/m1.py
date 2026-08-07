import ref

def check(workdir):
    from recomp.log_parser import parse_preempted_requests
    out = {"requests_matched": 0.0}
    try:
        got = parse_preempted_requests(ref.LOGS)
        want = ref.parse_preempted_requests(ref.LOGS)
        if got == want:
            out["requests_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"raised error: {type(e).__name__}: {str(e)[:120]}"
    return out
