def predict_fusion(node):
    if node.get("op") in ("MatMul", "Gemm") and node.get("has_scale") and node.get("axis") == 0:
        return "fusion"
    return "reformat"
