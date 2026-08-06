import ref


def check(workdir):
    from offload.policy import max_ngl_for_budget

    out = {"budget_matched": 0.0}
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for ngl in range(cfg["num_layers"] + 1):
            budget = ref.compute_vram(cfg, ngl)
            total += 1
            got = max_ngl_for_budget(cfg, budget)
            if got == ngl:
                ok += 1
    out["budget_matched"] = ok / total if total > 0 else 0.0
    return out
