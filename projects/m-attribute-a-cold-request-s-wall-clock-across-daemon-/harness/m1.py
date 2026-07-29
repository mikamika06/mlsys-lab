import ref


def check(workdir):
    from coldpath import build_phases

    out = {"cases_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, case in enumerate(ref.CASES):
        events = case["events"]
        want = ref.build_phases(events)
        got = build_phases(events)
        norm = [{k: v for k, v in p.items() if k in ("name", "start", "end", "duration_ms")}
                for p in (got or [])]
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {norm[:2]}, reference {want[:2]}"
    out["cases_matched"] = float(ok)
    return out
