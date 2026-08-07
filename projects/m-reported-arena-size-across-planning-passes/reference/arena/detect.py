def detect_dynamic_tensors(program):
    unplanned = []
    for node in program.get("nodes", []):
        for out in node.get("outputs", []):
            shape = out.get("shape", [])
            if any(isinstance(dim, str) or dim is None or dim < 0 for dim in shape):
                unplanned.append(out.get("name", "unknown"))
    return unplanned
