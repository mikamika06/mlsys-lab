def tracked_fusions(model_desc, level):
    fusions = []
    for n in model_desc.get("nodes", []):
        if level >= 1 and n.get("type") == "Conv" and n.get("has_bias"):
            fusions.append("ConvBiasFusion")
        if level >= 99 and n.get("type") == "MatMul":
            fusions.append("MatMulAddFusion")
    return sorted(list(set(fusions)))


def check_portability(model_desc):
    layout = model_desc.get("layout", "NCHW")
    target = model_desc.get("target", "generic")
    if layout == "NHWC" and target == "strict_nchw_only":
        return False
    return True
