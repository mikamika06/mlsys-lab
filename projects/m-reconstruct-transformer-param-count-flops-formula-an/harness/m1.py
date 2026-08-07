import ref

def check(workdir):
    from studentdesign.formulas import calculate_metrics
    out = {"formulas_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.compute_transformer_metrics(cfg)
        got = calculate_metrics(cfg)
        if abs(got["params"] - want["params"]) < 1e-5 and abs(got["flops"] - want["flops"]) < 1e-5:
            ok += 1
    out["formulas_matched"] = float(ok == len(ref.CONFIGS))
    return out
