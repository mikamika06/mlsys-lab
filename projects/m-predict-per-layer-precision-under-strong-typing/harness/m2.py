import ref

def check(workdir):
    from precision.sweep import run_precision_sweep
    out = {"sweep_matched": 0.0}
    try:
        formats = ["FP32", "FP16"]
        res = run_precision_sweep(ref.SWEEP_LAYERS[0], formats)
        if isinstance(res, dict) and "FP32" in res and "FP16" in res:
            if "mse" in res["FP16"] and "max_err" in res["FP16"]:
                out["sweep_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
