import ref


def check(workdir):
    from dyncomp.metrics import measure_ratios
    model, inputs = ref.get_test_model()
    try:
        res = measure_ratios(model, inputs)
    except Exception as e:
        return {"ratios_matched": 0.0, "_note": f"measure_ratios raised {type(e).__name__}: {str(e)[:100]}"}

    out = {"ratios_matched": 0.0}
    if isinstance(res, dict) and "compile_ratio" in res and "run_ratio" in res:
        if isinstance(res["compile_ratio"], (int, float)) and isinstance(res["run_ratio"], (int, float)):
            if res["compile_ratio"] > 0 and res["run_ratio"] > 0:
                out["ratios_matched"] = 1.0
            else:
                out["_note"] = "ratios must be positive numbers"
        else:
            out["_note"] = "ratios must be float or int values"
    else:
        out["_note"] = "measure_ratios must return a dict with compile_ratio and run_ratio"
    return out
