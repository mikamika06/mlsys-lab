import ref


def check(workdir):
    from finetune.predictor import predict_fits

    out = {"configs_matched": 0.0}
    want = ref.predict_fits(ref.CONFIGS, ref.VRAM_BUDGET)
    try:
        got = predict_fits(ref.CONFIGS, ref.VRAM_BUDGET)
    except Exception as e:
        out["_note"] = f"predictor raised {type(e).__name__}: {str(e)[:120]}"
        return out

    if not isinstance(got, list) or len(got) != len(want):
        out["_note"] = f"expected list of length {len(want)}, got {type(got)}"
        return out

    matched = 0
    for i, (w, g) in enumerate(zip(want, got)):
        if w.get("fits") == g.get("fits") and w.get("config_id") == g.get("config_id"):
            matched += 1
        elif i == 0:
            out["_note"] = f"mismatch at index 0: got {g}, reference {w}"

    out["configs_matched"] = float(matched)
    return out
