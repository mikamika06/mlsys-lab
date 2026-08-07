import ref

def check(workdir):
    from opt.transformer_fusion import apply_transformer_fusion
    from opt.validator import check_parity
    metrics = {"max_diff": 1.0, "cosine_sim": 0.0}
    g = ref.get_sample_graph()
    inputs = ref.get_sample_inputs()
    try:
        g_opt = apply_transformer_fusion(g)
        res = check_parity(g, g_opt, inputs)
        metrics["max_diff"] = float(res.get("max_diff", 1.0))
        metrics["cosine_sim"] = float(res.get("cosine_sim", 0.0))
    except Exception:
        pass
    return metrics
