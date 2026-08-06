import ref

def check(workdir):
    from ctinspect.scheme import infer_scheme_from_index

    out = {"schemes_matched": 0.0, "total": float(len(ref.INDEX_FIXTURES))}
    ok = 0
    for i, fixture in enumerate(ref.INDEX_FIXTURES):
        want = ref.infer_scheme_from_index(fixture)
        got = infer_scheme_from_index(fixture)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"fixture {i}: got {got}, reference {want}"

    out["schemes_matched"] = float(ok)
    return out
