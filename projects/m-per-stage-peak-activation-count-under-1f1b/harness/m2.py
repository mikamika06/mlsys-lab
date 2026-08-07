import ref

def check(workdir):
    from pact.trace import memory_trace
    out = {"trace_matched": 0.0}
    P, M = 4, 6
    want = ref.memory_trace(P, M)
    got = memory_trace(P, M)
    if got == want:
        out["trace_matched"] = 1.0
    else:
        out["_note"] = f"trace mismatch for P={P}, M={M}: got length {len(got) if isinstance(got, list) else type(got)}, want {len(want)}"
    return out
