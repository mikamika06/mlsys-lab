import ref

def check(workdir):
    from bertfuse.triage import triage_attention
    ok = 0
    for g in ref.GRAPHS:
        want = ref.triage_attention(g)
        got = triage_attention(g)
        if got == want:
            ok += 1
    return {"triage_matched": 1.0 if ok == len(ref.GRAPHS) else 0.0}
