import ref


def check(workdir):
    from tlog.metrics import compute_percentiles
    logs = ref.generate_logs(123)
    want = ref.ref_compute_percentiles(logs)
    try:
        got = compute_percentiles(logs)
    except Exception as e:
        return {"percentiles_match": 0.0, "_note": f"raised exception: {e}"}

    out = {"percentiles_match": 0.0}
    if not isinstance(got, dict):
        out["_note"] = "did not return a dict"
        return out

    keys = ["ttft_p50", "ttft_p99", "itl_p50", "itl_p99"]
    for k in keys:
        if k not in got:
            out["_note"] = f"missing key {k}"
            return out
        diff = abs(got[k] - want[k])
        if diff > 1e-5:
            out["_note"] = f"key {k} got {got[k]}, want {want[k]}"
            return out

    out["percentiles_match"] = 1.0
    return out
