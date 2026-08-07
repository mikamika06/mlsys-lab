MODELS = [
    {"nodes": [{"type": "Conv", "has_bias": True, "fusable": False}, {"type": "Add", "has_bias": False, "fusable": True}], "layout": "NCHW", "target": "generic"},
    {"nodes": [{"type": "MatMul", "has_bias": False, "fusable": False}, {"type": "Conv", "has_bias": True, "fusable": True}], "layout": "NHWC", "target": "strict_nchw_only"},
    {"nodes": [{"type": "Add", "has_bias": False, "fusable": True}, {"type": "Mul", "has_bias": False, "fusable": True}], "layout": "NCHW", "target": "edge"}
]

def count_nodes(model, level):
    nodes = list(model.get("nodes", []))
    base = len(nodes)
    if level == 0:
        return base
    if level == 1:
        fused = [n for n in nodes if not (n.get("type") in ("Add", "Mul") and n.get("fusable"))]
        return max(1, len(fused))
    if level >= 99:
        return max(1, base // 3)
    return base

def tracked_fusions(model, level):
    fusions = []
    for n in model.get("nodes", []):
        if level >= 1 and n.get("type") == "Conv" and n.get("has_bias"):
            fusions.append("ConvBiasFusion")
        if level >= 99 and n.get("type") == "MatMul":
            fusions.append("MatMulAddFusion")
    return sorted(list(set(fusions)))

def check_portability(model):
    layout = model.get("layout", "NCHW")
    target = model.get("target", "generic")
    if layout == "NHWC" and target == "strict_nchw_only":
        return False
    return True
