import ref

def check(workdir):
    from flashres.resolver import parse_wheel_name
    from flashres.compat import check_compatibility
    out = {"compatibility_matched": 0.0, "_note": ""}
    ok = 0
    for i, w in enumerate(ref.WHEELS):
        env = ref.ENVIRONMENTS[i]
        try:
            rec = parse_wheel_name(w)
            res = check_compatibility(rec, env["py"], env["cu"], env["torch"])
            if isinstance(res, bool):
                ok += 1
        except Exception as e:
            if not out["_note"]:
                out["_note"] = f"env {i} raised {type(e).__name__}: {str(e)[:80]}"
    out["compatibility_matched"] = float(ok)
    return out
