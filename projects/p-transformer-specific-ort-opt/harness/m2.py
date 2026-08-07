import ref

def check(workdir):
    from opt.transformer_fusion import apply_transformer_fusion
    from opt.analyzer import analyze_fused_nodes
    metrics = {"matched_patterns": 0.0, "node_mapping_valid": 0.0}
    g = ref.get_sample_graph()
    try:
        out_g = apply_transformer_fusion(g)
        analysis = analyze_fused_nodes(out_g)
        metrics["matched_patterns"] = float(analysis.get("matched_patterns", 0))
        metrics["node_mapping_valid"] = float(analysis.get("node_mapping_valid", 0))
    except Exception:
        pass
    return metrics
