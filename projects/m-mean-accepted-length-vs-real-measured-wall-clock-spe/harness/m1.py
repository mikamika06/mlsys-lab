import ref


def check(workdir):
    from specprof.metrics import calculate_metrics

    out = {"metrics_matched": 0.0, "total": float(len(ref.MOCK_RUNS))}
    ok = 0
    for i, run in enumerate(ref.MOCK_RUNS):
        want = ref.calculate_metrics(run)
        got = calculate_metrics(run)

        matched = True
        for key in [
            "mean_accepted_length",
            "real_speedup",
            "theoretical_speedup",
            "overhead_ratio",
        ]:
            if abs(got.get(key, 0.0) - want[key]) > 1e-4:
                matched = False
                if "_note" not in out:
                    out["_note"] = (
                        f"run {i} metric {key}: got {got.get(key)}, want {want[key]}"
                    )
                break
        if matched:
            ok += 1

    out["metrics_matched"] = float(ok)
    return out
