def detect_baked_constants(model_spec):
    count = 0
    nodes = model_spec.get("nodes", [])
    for node in nodes:
        if node.get("op") == "Constant" and node.get("is_baked", True):
            count += 1
    return count
