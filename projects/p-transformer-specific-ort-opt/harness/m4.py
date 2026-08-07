import ref

def check(workdir):
    from opt.transformer_fusion import apply_transformer_fusion
    from opt.analyzer import measure_phases
    metrics = {"prefill_speedup": 0.0, "decode_speedup": 0.0}
    g = ref.get_sample_graph()
    inputs = ref.get_sample_inputs()
    try:
        g_opt = apply_transformer_fusion(g)
        res = measure_phases(g, g_opt, inputs)
        metrics["prefill_speedup"] = float(res.get("prefill_speedup", 0.0))
        metrics["decode_speedup"] = float(res.get("decode_speedup", 0.0))
    except Exception:
        pass
    return metrics
