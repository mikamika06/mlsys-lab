import ref


def check(workdir):
    from ttft.model import fit_latency_model, predict_ttft
    from ttft.analyze import compute_relative_error

    out = {"rel_err_match": 0.0, "parameters_valid": 0.0}
    try:
        lengths = [x[0] for x in ref.RAW_DATA]
        times = [x[1] for x in ref.RAW_DATA]
        params = fit_latency_model(lengths, times)

        if isinstance(params, dict) and "slope" in params and "intercept" in params:
            out["parameters_valid"] = 1.0
            preds = [predict_ttft(params, l) for l in lengths]
            err = compute_relative_error(times, preds)
            if err < 0.5:
                out["rel_err_match"] = 1.0
            else:
                out["_note"] = f"Relative error too high: {err}"
        else:
            out["_note"] = "Parameters format invalid"
    except Exception as e:
        out["_note"] = f"Error in model execution: {type(e).__name__}: {str(e)[:120]}"
    return out
