import ref

def check(workdir):
    from hpa.simulator import simulate_hpa, tune_hpa

    out = {"simulate_match": 0.0, "tune_match": 0.0}

    try:
        got = simulate_hpa(ref.METRICS_1, 5, 10.0, 3)
        want = ref.simulate_hpa(ref.METRICS_1, 5, 10.0, 3)
        if got == want:
            out["simulate_match"] = 1.0
        else:
            out["_note"] = f"simulate_hpa mismatch. Got {got[:5]}, want {want[:5]}"

        best_got = tune_hpa(ref.METRICS_1, 5, ref.CONFIGS_1)
        best_want = ref.tune_hpa(ref.METRICS_1, 5, ref.CONFIGS_1)
        if best_got == best_want:
            out["tune_match"] = 1.0
        else:
            out["_note"] = out.get("_note", "") + f" tune_hpa mismatch. Got {best_got}, want {best_want}"
    except Exception as e:
        out["_note"] = str(e)

    return out
