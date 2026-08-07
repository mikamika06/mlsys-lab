def identify_intermediate_opset(nodes, graph_opset_map=None):
    """Identify effective intermediate ONNX opset version across nodes and subgraphs."""
    if graph_opset_map is None:
        graph_opset_map = {}
    node_opsets = []
    for node in nodes:
        domain = node.get("domain", "ai.onnx")
        version = node.get("version")
        if version is None:
            version = graph_opset_map.get(domain, 13)
        node_opsets.append(int(version))
    if not node_opsets:
        return 13
    return max(node_opsets)


def validate_opset_compatibility(opset_version, required_min=13, required_max=19):
    """Check if opset version meets target OpenVINO/NNCF requirements."""
    version = int(opset_version)
    is_valid = required_min <= version <= required_max
    return {
        "version": version,
        "is_valid": is_valid,
        "recommended_version": min(max(version, required_min), required_max),
    }
