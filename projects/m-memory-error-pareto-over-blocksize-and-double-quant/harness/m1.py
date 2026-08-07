import ref


def check(workdir):
    from quantplan.pareto import compute_pareto

    out = {"pareto_matched": 0.0}
    cfg = ref.CONFIGS[0]
    want = ref.compute_pareto(cfg["weights_shape"], cfg["blocksizes"], cfg["double_quants"])
    try:
        got = compute_pareto(cfg["weights_shape"], cfg["blocksizes"], cfg["double_quants"])
        if len(got) == len(want):
            matched = True
            for g, w in zip(got, want):
                if g.get("blocksize") != w["blocksize"] or g.get("double_quant") != w["double_quant"]:
                    matched = False
                    break
                if abs(g.get("memory_bytes", -1) - w["memory_bytes"]) > 1 or abs(g.get("mse", -1) - w["mse"]) > 1e-5:
                    matched = False
                    break
            if matched:
                out["pareto_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"error in compute_pareto: {e}"
    return out
