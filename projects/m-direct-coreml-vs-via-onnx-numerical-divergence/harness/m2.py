import ref

def check(workdir):
    from coreml_audit import census
    out = {"census_match": 0.0}
    ok = 0
    for i, m in enumerate(ref.MODELS):
        want = ref.compute_census(m)
        got = census.node_census(m["nodes"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, reference {want}"
    if ok == len(ref.MODELS):
        out["census_match"] = 1.0
    return out
