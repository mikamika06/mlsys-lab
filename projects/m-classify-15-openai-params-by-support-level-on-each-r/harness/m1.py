import ref


def check(workdir):
    from oaicompat import classify_params

    out = {"runners_matched": 0.0, "runners": float(len(ref.RUNNERS))}
    ok = 0
    for runner in ref.RUNNERS:
        want = sorted(ref.classify_params(runner), key=lambda r: r["param"])
        got_raw = classify_params(runner)
        norm = sorted(
            [{k: v for k, v in row.items() if k in ("param", "level")}
             for row in (got_raw or [])],
            key=lambda r: r.get("param", ""),
        )
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"runner {runner['name']}: got {norm[:3]}, reference {want[:3]}"
    out["runners_matched"] = float(ok)
    return out
