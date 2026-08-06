import ref


def check(workdir):
    from distmem.traces import classify_trace_pattern

    ok = 0
    for case in ref.TRACE_CASES:
        want = ref.ref_classify_trace(case["events"])
        got = classify_trace_pattern(case["events"])
        if got == want:
            ok += 1
    return {"traces_classified": 1.0 if ok == len(ref.TRACE_CASES) else 0.0}
