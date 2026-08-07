import ref


def check(workdir):
    from vllmbench.metrics import compute_percentiles

    raw = ref.get_raw_dataset()
    want = ref.compute_reference_percentiles(raw)
    try:
        got = compute_percentiles(raw)
    except Exception as e:
        return {"percentiles_matched": 0.0, "_note": f"raised exception: {e}"}

    matched = 1.0
    for k in ["ttft", "tpot", "itl", "e2el"]:
        if k not in got:
            matched = 0.0
            break
        for subk in ["mean", "p50", "p99"]:
            if subk not in got[k] or abs(got[k][subk] - want[k][subk]) > 1e-5:
                matched = 0.0

    return {"percentiles_matched": matched}
