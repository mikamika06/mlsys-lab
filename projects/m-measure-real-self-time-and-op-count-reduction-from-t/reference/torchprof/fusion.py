def count_fused_kernels(trace_nodes):
    fused = 0
    unfused = 0
    for node in trace_nodes:
        name = node.get("name", "")
        if "fused" in name or "fusion" in name or node.get("is_fused", False):
            fused += 1
        else:
            unfused += 1
    return {"fused": fused, "unfused": unfused}
