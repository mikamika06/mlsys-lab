"""CUDA Graph legality classifier implementation."""


def is_graph_legal(node_spec):
    op_type = node_spec.get("op_type", "")
    if op_type in ("CpuFallbackOp", "CustomHostOp", "DynamicShapeAlloc"):
        return False, "disallowed_op"
    if node_spec.get("has_control_flow", False):
        return False, "control_flow"
    if node_spec.get("allocates_host_pinned", False):
        return False, "host_pinned_alloc"
    if node_spec.get("dynamic_resizing", False):
        return False, "dynamic_shape"
    if not node_spec.get("aligned_io", True):
        return False, "unaligned_io"
    return True, "ok"


def classify_pipeline(nodes):
    legal_count = 0
    reasons = []
    for node in nodes:
        legal, reason = is_graph_legal(node)
        if legal:
            legal_count += 1
        else:
            reasons.append(reason)
    return {
        "is_legal": legal_count == len(nodes),
        "legal_nodes": legal_count,
        "total_nodes": len(nodes),
        "reasons": reasons,
    }
