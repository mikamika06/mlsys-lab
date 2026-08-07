PROGRAMS = [
    {
        "metadata": {"version": 1},
        "nodes": [
            {"name": "a", "op": "placeholder", "target": "a", "inputs": [], "outputs": [{"name": "a_out", "shape": [4, 4], "dtype": "float32"}]},
            {"name": "b", "op": "placeholder", "target": "b", "inputs": [], "outputs": [{"name": "b_out", "shape": [4, 4], "dtype": "float32"}]},
            {"name": "mul", "op": "call_function", "target": "mul", "inputs": ["a_out", "b_out"], "outputs": [{"name": "mul_out", "shape": [4, 4], "dtype": "float32"}]},
            {"name": "out", "op": "output", "target": "output", "inputs": ["mul_out"], "outputs": []}
        ]
    },
    {
        "metadata": {"version": 2},
        "nodes": [
            {"name": "inp", "op": "placeholder", "target": "inp", "inputs": [], "outputs": [{"name": "inp_out", "shape": [8], "dtype": "int64"}]},
            {"name": "relu", "op": "call_function", "target": "relu", "inputs": ["inp_out"], "outputs": [{"name": "relu_out", "shape": [8], "dtype": "int64"}]},
            {"name": "out", "op": "output", "target": "output", "inputs": ["relu_out"], "outputs": []}
        ]
    },
    {
        "metadata": {"version": 3},
        "nodes": [
            {"name": "t", "op": "placeholder", "target": "t", "inputs": [], "outputs": [{"name": "t_out", "shape": [1, 3, 224, 224], "dtype": "float32"}]},
            {"name": "noop", "op": "noop", "target": "noop", "inputs": [], "outputs": []},
            {"name": "out", "op": "output", "target": "output", "inputs": ["t_out"], "outputs": []}
        ]
    }
]


def parse_graph(prog):
    nodes = []
    for node in prog.get("nodes", []):
        nodes.append({
            "name": node.get("name"),
            "op": node.get("op"),
            "target": node.get("target"),
            "inputs": list(node.get("inputs", [])),
            "outputs": list(node.get("outputs", []))
        })
    return {"nodes": nodes, "metadata": prog.get("metadata", {})}


def inspect_shapes(parsed):
    shapes = {}
    for node in parsed.get("nodes", []):
        for out in node.get("outputs", []):
            if "shape" in out:
                shapes[out["name"]] = tuple(out["shape"])
    return shapes


def inspect_dtypes(parsed):
    dtypes = {}
    for node in parsed.get("nodes", []):
        for out in node.get("outputs", []):
            if "dtype" in out:
                dtypes[out["name"]] = out["dtype"]
    return dtypes


def inspect_io(parsed):
    inputs = []
    outputs = []
    for node in parsed.get("nodes", []):
        if node.get("op") == "placeholder":
            inputs.append(node.get("name"))
        elif node.get("op") == "output":
            outputs.extend(node.get("inputs", []))
    return {"inputs": inputs, "outputs": outputs}


def optimize_graph(parsed):
    nodes = parsed.get("nodes", [])
    filtered = [n for n in nodes if n.get("op") != "noop"]
    return {"nodes": filtered, "metadata": parsed.get("metadata", {})}
