def audit_bindings(onnx_graph, registered_plugins):
    unbound = []
    for node in onnx_graph.get("nodes", []):
        domain = node.get("domain", "ai.onnx")
        op_type = node.get("op_type")
        if domain != "ai.onnx":
            key = f"{domain}::{op_type}"
            if key not in registered_plugins:
                unbound.append(node.get("name", op_type))
    return {
        "total_custom_nodes": len([n for n in onnx_graph.get("nodes", []) if n.get("domain", "ai.onnx") != "ai.onnx"]),
        "unbound_nodes": unbound,
        "is_fully_bound": len(unbound) == 0,
    }
