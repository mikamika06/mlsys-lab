import ref


def check(workdir):
    from launchgraph.predict import predict_speedup

    out = {"predictions_matched": 0.0}
    ok = 0
    total = len(ref.PREDICT_CONFIGS)

    for i, cfg in enumerate(ref.PREDICT_CONFIGS):
        want = ref.predict_speedup(**cfg)
        try:
            got = predict_speedup(**cfg)
            if abs(got - want) < 1e-5:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"cfg {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"cfg {i} raised {type(e).__name__}: {e}"

    if ok == total:
        out["predictions_matched"] = 1.0

    return out
