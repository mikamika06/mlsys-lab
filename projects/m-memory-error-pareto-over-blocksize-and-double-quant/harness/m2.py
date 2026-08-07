import ref


def check(workdir):
    from quantplan.budget import best_config

    out = {"budget_matched": 0.0}
    cfg = ref.CONFIGS[0]
    want = ref.best_config(cfg["weights_shape"], cfg["blocksizes"], cfg["double_quants"], cfg["mse_budget"])
    try:
        got = best_config(cfg["weights_shape"], cfg["blocksizes"], cfg["double_quants"], cfg["mse_budget"])
        if got and got.get("blocksize") == want["blocksize"] and got.get("double_quant") == want["double_quant"]:
            out["budget_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error in best_config: {e}"
    return out
