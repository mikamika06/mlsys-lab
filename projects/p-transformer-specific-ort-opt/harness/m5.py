import ref

def check(workdir):
    from opt.transformer_fusion import apply_transformer_fusion
    from opt.analyzer import measure_phases
    from opt.validator import check_parity
    metrics = {"overall_speedup": 0.0, "parity_ok": 0.0}
    g = ref.get_sample_graph()
    inputs = ref.get_sample_inputs()
    try:
        g_opt = apply_transformer_fusion(g)
        res_perf = measure_phases(g, g_opt, inputs)
        res_par = check_parity(g, g_opt, inputs)
        metrics["overall_speedup"] = float(res_perf.get("overall_speedup", 0.0))
        metrics["parity_ok"] = float(res_par.get("parity_ok", 0.0))
    except Exception:
        pass
    return metrics
