import ref


def check(workdir):
    from rundiag import repetition_report

    out = {"cases_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, (tokens, window, threshold) in enumerate(ref.CASES):
        want = ref.repetition_report(list(tokens), window, threshold)
        got = repetition_report(list(tokens), window, threshold)
        norm = _normalize(got)
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {norm}, reference {want}"
    out["cases_matched"] = float(ok)
    return out


def _normalize(report):
    if not isinstance(report, dict):
        return report
    keys = ("triggered", "token", "window_count", "positions", "histogram", "total_tokens", "unique_tokens")
    norm = {}
    for k in keys:
        v = report.get(k)
        if k == "positions" and isinstance(v, list):
            v = sorted(v)
        norm[k] = v
    return norm
