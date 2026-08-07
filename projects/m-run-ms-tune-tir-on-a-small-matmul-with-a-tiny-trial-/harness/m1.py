import ref


def check(workdir):
    out = {"latency_extracted": 0.0}
    try:
        from tuntir.tune import tune_small_matmul
        lat = tune_small_matmul(workdir)
        if isinstance(lat, (int, float)) and lat > 0:
            out["latency_extracted"] = 1.0
        else:
            out["_note"] = f"invalid latency value: {lat}"
    except Exception as e:
        out["_note"] = f"exception during tune_small_matmul: {type(e).__name__}: {e}"
    return out
