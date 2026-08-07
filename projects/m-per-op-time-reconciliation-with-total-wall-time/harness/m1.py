import ref


def check(workdir):
    from reconcile.profiler import reconcile_profile_times

    out = {"reports_matched": 0.0, "total": float(len(ref.REPORTS))}
    ok = 0
    for i, report in enumerate(ref.REPORTS):
        want = ref.reconcile_profile_times(report)
        try:
            got = reconcile_profile_times(report)
        except Exception as e:
            out["_note"] = f"report {i} raised {type(e).__name__}: {str(e)}"
            break

        diff = abs(got.get("overhead_ratio", 1.0) - want["overhead_ratio"])
        if diff < 1e-4 and got.get("reconciled") == want["reconciled"]:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"report {i}: got {got}, want {want}"

    out["reports_matched"] = float(ok)
    return out
