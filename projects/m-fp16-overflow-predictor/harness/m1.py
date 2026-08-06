import ref

def check(workdir):
    from fp16pred.predictor import predict_overflow
    out = {"predictions_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.predict_overflow(cfg)
        got = predict_overflow(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["predictions_matched"] = float(ok)
    return out
