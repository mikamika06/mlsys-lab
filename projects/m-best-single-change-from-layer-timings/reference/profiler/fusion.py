def fused_layer_count(raw_onnx_str, simplified_onnx_str):
    raw_nodes = raw_onnx_str.count("node")
    simp_nodes = simplified_onnx_str.count("node")
    return {"raw_nodes": raw_nodes, "simplified_nodes": simp_nodes, "fused_count": max(0, raw_nodes - simp_nodes)}
