import ref


def check(workdir):
    from throttling.detect import detect_transition
    from throttling.metrics import severity_score

    traces = ref.TRACES
    sev_match = 0
    ratio_match = 0
    out = {"severity_match": 0.0, "ratio_match": 0.0}
    for trace, _, expected_drop in traces:
        trans = detect_transition(trace)
        got_sev = severity_score(trace, trans)
        want_sev = ref.compute_severity(trace, trans)
        if abs(got_sev - want_sev) < 1e-2:
            sev_match += 1
        if abs(got_sev - expected_drop) < 0.15:
            ratio_match += 1

    out["severity_match"] = 1.0 if sev_match == len(traces) else 0.0
    out["ratio_match"] = 1.0 if ratio_match == len(traces) else 0.0
    return out
