import ref

def check(workdir):
    from skew.parser import parse_metadata
    out = {"parsed_matched": 0.0}
    ok = 0
    for i, h in enumerate(ref.HEADERS):
        want = ref.parse_metadata(h)
        got = parse_metadata(h)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"header {i}: got {got}, reference {want}"
    out["parsed_matched"] = float(ok)
    return out
