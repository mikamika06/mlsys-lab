def inspect_shapes(parsed_graph):
    shapes = {}
    for node in parsed_graph.get("nodes", []):
        for out in node.get("outputs", []):
            if "shape" in out:
                shapes[out["name"]] = tuple(out["shape"])
    return shapes


def inspect_dtypes(parsed_graph):
    dtypes = {}
    for node in parsed_graph.get("nodes", []):
        for out in node.get("outputs", []):
            if "dtype" in out:
                dtypes[out["name"]] = out["dtype"]
    return dtypes


def inspect_io(parsed_graph):
    inputs = []
    outputs = []
    for node in parsed_graph.get("nodes", []):
        if node.get("op") == "placeholder":
            inputs.append(node.get("name"))
        elif node.get("op") == "output":
            outputs.extend(node.get("inputs", []))
    return {"inputs": inputs, "outputs": outputs}
