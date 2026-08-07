import ref

def check(workdir):
    from flashres.resolver import parse_wheel_name
    out = {"wheels_parsed": 0.0, "_note": ""}
    ok = 0
    for i, w in enumerate(ref.WHEELS):
        try:
            got = parse_wheel_name(w)
            if isinstance(got, dict) and "distribution" in got and "version" in got and "py_tag" in got:
                ok += 1
        except Exception as e:
            if not out["_note"]:
                out["_note"] = f"wheel {i} raised {type(e).__name__}: {str(e)[:80]}"
    out["wheels_parsed"] = float(ok)
    return out
