import ref


def check(workdir):
    out = {"thrash_match": 0.0, "traces_match": 0.0}

    try:
        from vllm_policy.logs import classify_traces, detect_thrash
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    t_ok = 0
    for logs, args in ref.LOG_CASES:
        try:
            want = ref.detect_thrash(logs, args)
            got = detect_thrash(logs, args)
            if got == want:
                t_ok += 1
        except Exception:
            pass

    if ref.LOG_CASES:
        out["thrash_match"] = float(t_ok) / len(ref.LOG_CASES)

    try:
        want_tr = ref.classify_traces(ref.TRACE_CASES)
        got_tr = classify_traces(ref.TRACE_CASES)
        if got_tr == want_tr:
            out["traces_match"] = 1.0
    except Exception:
        pass

    return out
