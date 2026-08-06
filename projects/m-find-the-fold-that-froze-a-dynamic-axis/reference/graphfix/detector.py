def find_frozen_folds(graph):
    frozen = []
    symbolic_inputs = {inp["name"] for inp in graph.get("inputs", []) if any(isinstance(d, str) for d in inp.get("shape", []))}
    for node in graph.get("nodes", []):
        if node.get("op") in ("ConstantFold", "Shape", "Gather", "Reshape"):
            for inp in node.get("inputs", []):
                if inp in symbolic_inputs or node.get("attributes", {}).get("forces_static", False):
                    frozen.append(node["name"])
                    break
    return sorted(list(set(frozen)))
