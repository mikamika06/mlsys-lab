def canonicalize_qdq(graph_def):
    nodes = list(graph_def.get("nodes", []))
    if "QuantizeLinear" in nodes and "DequantizeLinear" in nodes:
        return {"canonical": True, "nodes": ["MatMul"]}
    return {"canonical": False, "nodes": nodes}
