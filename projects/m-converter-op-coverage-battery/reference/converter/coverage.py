def audit_op_coverage(graph_spec, runtime_target):
    supported_ops = set(runtime_target.get("supported_ops", []))
    decomposable_ops = set(runtime_target.get("decomposable_ops", {}).keys())

    node_status = {}
    supported_count = 0
    decomposable_count = 0
    unsupported_count = 0

    for node in graph_spec.get("nodes", []):
        op_type = node["op_type"]
        node_id = node["id"]
        if op_type in supported_ops:
            status = "NATIVE"
            supported_count += 1
        elif op_type in decomposable_ops:
            status = "DECOMPOSABLE"
            decomposable_count += 1
        else:
            status = "UNSUPPORTED"
            unsupported_count += 1
        node_status[node_id] = status

    total = len(graph_spec.get("nodes", []))
    ratio = (supported_count + decomposable_count) / total if total > 0 else 0.0

    return {
        "node_status": node_status,
        "supported_count": supported_count,
        "decomposable_count": decomposable_count,
        "unsupported_count": unsupported_count,
        "coverage_ratio": ratio
    }
