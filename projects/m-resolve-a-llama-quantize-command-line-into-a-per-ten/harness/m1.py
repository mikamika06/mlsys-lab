import ref

def check(workdir):
    from lquant.plan import resolve_plan
    from lquant.overrides import parse_overrides

    out = {"plans_matched": 0.0, "total": float(len(ref.MODELS))}
    ok = 0
    for i, m in enumerate(ref.MODELS):
        ov = parse_overrides(m["args"][1:]) if len(m["args"]) > 1 else {}
        want = resolve_plan(m["tensors"], m["default"], m["overrides"])
        try:
            got = resolve_plan(m["tensors"], m["default"], ov)
        except Exception:
            got = {}
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, want {want}"
    out["plans_matched"] = float(ok)
    return out
