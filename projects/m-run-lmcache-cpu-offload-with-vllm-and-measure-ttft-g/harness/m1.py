import ref


def check(workdir):
    from offload.runner import measure_ttft_gain

    out = {"latency_ratios_matched": 0.0}
    ok = 0
    for reqs, cfg in zip(ref.REQUESTS_SET, ref.CONFIGS_SET):
        want = ref.measure_ttft_gain(reqs, cfg)
        try:
            got = measure_ttft_gain(reqs, cfg)
        except Exception as e:
            out["_note"] = f"runner raised {type(e).__name__}: {e}"
            return out
        if isinstance(got, dict) and "mean_ratio" in got:
            if abs(got["mean_ratio"] - want["mean_ratio"]) < 1e-5:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"got mean_ratio {got.get('mean_ratio')}, want {want['mean_ratio']}"
        elif "_note" not in out:
            out["_note"] = f"invalid return format: {got}"
    out["latency_ratios_matched"] = float(ok)
    return out
