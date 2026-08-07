import ref


def check(workdir):
    from quant.size import actual_flatbuffer_size, predict_quantized_size

    out = {"models_matched": 0.0, "rel_err": 1.0}
    ok = 0
    max_err = 0.0
    for model in ref.MODELS:
        for mode in ("int8", "int16x8", "dynamic_range"):
            want_pred = ref.predict_quantized_size(model, mode)
            want_act = ref.actual_flatbuffer_size(model, mode)
            got_pred = predict_quantized_size(model, mode)
            got_act = actual_flatbuffer_size(model, mode)

            err1 = abs(got_pred - want_pred) / float(want_pred) if want_pred > 0 else 0.0
            err2 = abs(got_act - want_act) / float(want_act) if want_act > 0 else 0.0
            cur_err = max(err1, err2)
            if cur_err > max_err:
                max_err = cur_err
            if got_pred == want_pred and got_act == want_act:
                ok += 1

    out["models_matched"] = float(ok)
    out["rel_err"] = float(max_err)
    return out
