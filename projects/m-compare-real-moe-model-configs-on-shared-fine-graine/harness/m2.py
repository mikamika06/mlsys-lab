import ref


def check(workdir):
    from moecomp.metrics import compute_costs
    from moecomp.reporter import compare_configs

    out = {"rel_err_match": 0.0, "param_match": 0.0, "ratio_match": 0.0}
    try:
        ok_costs = 0
        for cfg in ref.CONFIGS:
            want_c = ref.compute_costs(cfg)
            got_c = compute_costs(cfg)
            if got_c == want_c:
                ok_costs += 1

        if ok_costs == len(ref.CONFIGS):
            out["param_match"] = 1.0

        c1 = ref.compute_costs(ref.CONFIGS[0])
        c2 = ref.compute_costs(ref.CONFIGS[1])
        want_ratio = c1["total_params"] / max(1, c2["total_params"])

        rep = compare_configs(ref.CONFIGS[0], ref.CONFIGS[1])
        got_ratio = rep.get("ratio", 0.0)

        if abs(got_ratio - want_ratio) < 1e-5:
            out["ratio_match"] = 1.0
            out["rel_err_match"] = 1.0

    except Exception as e:
        out["_note"] = f"error during metric evaluation: {type(e).__name__}: {str(e)[:100]}"

    return out
