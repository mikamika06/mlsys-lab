import ref


def check(workdir):
    from latency.stats import compute_robust_stats

    out = {"stats_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_robust_stats(cfg["samples"])
        got = compute_robust_stats(cfg["samples"])
        if isinstance(got, dict) and all(k in got for k in ("median", "mad", "p95")):
            if (abs(got["median"] - want["median"]) < 1e-4 and
                abs(got["mad"] - want["mad"]) < 1e-4 and
                abs(got["p95"] - want["p95"]) < 1e-4):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, reference {want}"
        elif "_note" not in out:
            out["_note"] = f"config {i}: invalid return format {got}"

    out["stats_matched"] = float(ok)
    return out
