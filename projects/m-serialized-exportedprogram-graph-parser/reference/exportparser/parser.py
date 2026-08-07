def parse_exported_program(serialized_dict):
    """Parse a serialized ExportedProgram dict into a structured graph IR dict."""
    nodes = []
    for raw in serialized_dict.get("nodes", []):
        node = {
            "name": str(raw["name"]),
            "op": str(raw["op"]),
            "target": str(raw.get("target", "")),
            "args": list(raw.get("args", [])),
            "kwargs": dict(raw.get("kwargs", {})),
            "side_effect": bool(raw.get("side_effect", False)),
        }
        nodes.append(node)
    return {
        "inputs": list(serialized_dict.get("inputs", [])),
        "outputs": list(serialized_dict.get("outputs", [])),
        "nodes": nodes,
    }


def validate_schema(graph_ir):
    """Validate graph inputs, outputs, and node references."""
    known = set(graph_ir.get("inputs", []))
    for node in graph_ir.get("nodes", []):
        known.add(node["name"])

    for node in graph_ir.get("nodes", []):
        for arg in node.get("args", []):
            if isinstance(arg, str) and arg.startswith("%") and arg[1:] not in known:
                return False
        for v in node.get("kwargs", {}).values():
            if isinstance(v, str) and v.startswith("%") and v[1:] not in known:
                return False

    for out in graph_ir.get("outputs", []):
        if out not in known:
            return False

    return True
