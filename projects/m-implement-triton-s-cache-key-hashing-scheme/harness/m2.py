import ref


def check(workdir):
    from tcache.metrics import compute_metrics

    out = {"metrics_matched": 0.0}
    ok = 0
    for counters in ref.COUNTERS:
        want = ref.compute_metrics(counters)
        got = compute_metrics(counters)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"counters {counters}: got {got}, want {want}"
    out["metrics_matched"] = 1.0 if ok == len(ref.COUNTERS) else 0.0
    return out
