def find_frozen_dynamic_folds(graph):
    dynamic_tensors = set()
    for inp in graph.get("inputs", []):
        shape = inp.get("shape", [])
        for dim in shape:
            if isinstance(dim, str):
                dynamic_tensors.add(inp["name"])
                break

    tensor_producers = {}
    for node in graph.get("nodes", []):
        for out in node.get("outputs", []):
            tensor_producers[out] = node

    propagated_dynamic = set(dynamic_tensors)
    changed = True
    while changed:
        changed = False
        for node in graph.get("nodes", []):
            if any(inp in propagated_dynamic for inp in node.get("inputs", [])):
                for out in node.get("outputs", []):
                    if out not in propagated_dynamic:
                        propagated_dynamic.add(out)
                        changed = True

    frozen_folds = []
    for node in graph.get("nodes", []):
        if node.get("op") in ("ConstantFold", "Shape", "Gather", "Reshape", "Slice"):
            has_dynamic_input = any(inp in propagated_dynamic for inp in node.get("inputs", []))
            is_constant_output = node.get("is_folded", False) or node.get("static_shape", False)
            if has_dynamic_input and is_constant_output:
                frozen_folds.append(node["name"])

    return sorted(frozen_folds)
