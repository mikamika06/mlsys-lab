import ref


def check(workdir):
    from opside import classify_all

    out = {"cases_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, cfg in enumerate(ref.CASES):
        want = ref.classify_all(cfg)
        got = classify_all(cfg)
        norm = [{k: v for k, v in item.items() if k in ("name", "side")}
                for item in (got or [])]
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {norm[:3]}, reference {want[:3]}"
    out["cases_matched"] = float(ok)
    return out
