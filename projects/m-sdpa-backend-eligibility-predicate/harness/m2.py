import ref


def check(workdir):
    from sdpa_pred.predictor import predict_backend

    out = {"predictions_matched": 0.0}
    ok = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        want = ref.predict_backend(cfg["dtype"], cfg["is_causal"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["device_cap"])
        got = predict_backend(cfg["dtype"], cfg["is_causal"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["device_cap"])
        if want == got:
            ok += 1
    if ok == total:
        out["predictions_matched"] = 1.0
    else:
        out["_note"] = f"matched {ok}/{total} backend predictions"
    return out
