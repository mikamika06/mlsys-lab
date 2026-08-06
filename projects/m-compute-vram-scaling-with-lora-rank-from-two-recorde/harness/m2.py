import ref


def check(workdir):
    from lorascaling.extractor import extract_rank_scaling
    from lorascaling.predictor import predict_rank_requirements

    out = {"rel_err": 1.0}
    max_err = 0.0

    for i, (log1, log2, target_rank) in enumerate(ref.LOG_PAIRS):
        want_params = ref.oracle_extract(log1, log2)
        want_pred = ref.oracle_predict(want_params, target_rank)

        try:
            got_params = extract_rank_scaling(log1, log2)
            got_pred = predict_rank_requirements(got_params, target_rank)

            err_vram = abs(got_pred["predicted_vram_bytes"] - want_pred["predicted_vram_bytes"]) / want_pred["predicted_vram_bytes"]
            err_flops = abs(got_pred["predicted_step_flops"] - want_pred["predicted_step_flops"]) / want_pred["predicted_step_flops"]

            max_err = max(max_err, err_vram, err_flops)
        except Exception as e:
            out["_note"] = f"pair {i} prediction raised exception: {type(e).__name__}: {str(e)}"
            return out

    out["rel_err"] = float(max_err)
    return out
