import ref


def check(workdir):
    from throttling.detect import detect_transition

    traces = ref.TRACES
    ok = 0
    out = {"transitions_matched": 0.0}
    for i, (trace, expected_transition, _) in enumerate(traces):
        got = detect_transition(trace)
        if abs(got - expected_transition) <= 15:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got}, reference {expected_transition}"
    out["transitions_matched"] = float(ok)
    return out
