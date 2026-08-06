import ref


def check(workdir):
    from frag.auditor import audit_block_trace

    out = {"audit_match": 0.0}
    ok = True
    for i, trace in enumerate(ref.TRACES):
        want = ref.audit_block_trace(trace)
        got = audit_block_trace(trace)
        if got != want:
            ok = False
            out["_note"] = f"trace {i}: got {got}, want {want}"
            break

    if ok:
        out["audit_match"] = 1.0
    return out
