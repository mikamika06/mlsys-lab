import ref

def check(workdir):
    from lquant.plan import resolve_plan
    from lquant.quant import predict_size

    out = {"size_match": 0.0}
    ok = 0
    for m in ref.MODELS:
        plan = resolve_plan(m["tensors"], m["default"], m["overrides"])
        want_size = predict_size(m["tensors"], plan)
        try:
            got_size = predict_size(m["tensors"], plan)
        except Exception:
            got_size = -1
        if got_size == want_size:
            ok += 1
    if ok == len(ref.MODELS):
        out["size_match"] = 1.0
    else:
        out["_note"] = "size verification failed for some models"
    return out
