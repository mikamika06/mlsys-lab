import ref

def check(workdir):
    from opt.transformer_fusion import apply_transformer_fusion
    metrics = {"pass_applied": 0.0, "nodes_fused_count": 0.0}
    g = ref.get_sample_graph()
    try:
        out = apply_transformer_fusion(g)
        if isinstance(out, dict) and "nodes" in out:
            metrics["pass_applied"] = 1.0
            metrics["nodes_fused_count"] = float(out.get("fused_count", 0))
    except Exception:
        pass
    return metrics
