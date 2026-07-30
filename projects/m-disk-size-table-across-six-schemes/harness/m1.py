import ref


def _matches(got, want):
    if not isinstance(got, list) or len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if not isinstance(g, dict):
            return False
        if g.get("scheme") != w["scheme"] or g.get("bits") != w["bits"] or g.get("bytes") != w["bytes"]:
            return False
        try:
            if abs(float(g.get("ratio")) - w["ratio"]) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def check(workdir):
    from diskplan import size_table

    out = {"size_ratio": 0.0, "cases": float(len(ref.CONFIGS))}
    ok = 0
    for i, (model, schemes) in enumerate(ref.CONFIGS):
        want = ref.size_table(model, schemes)
        try:
            got = size_table(model, schemes)
        except Exception as e:
            got = None
            if "_note" not in out:
                out["_note"] = f"case {i}: raised {type(e).__name__}: {str(e)[:120]}"
        if got is not None and _matches(got, want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got[:2] if isinstance(got, list) else got}, reference {want[:2]}"
    out["size_ratio"] = ok / len(ref.CONFIGS)
    return out
