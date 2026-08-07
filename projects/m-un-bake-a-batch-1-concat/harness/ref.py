GRAPHS = [
    {"node": [{"op": "Concat", "attribute": [{"name": "axis", "i": 0}]}]},
    {"node": [{"op": "Concat", "attribute": [{"name": "axis", "i": 1}]}]},
    {"node": [{"op": "Add", "attribute": []}, {"op": "Concat", "attribute": [{"name": "axis", "i": 0}]}]}
]

MODELS = [
    {"input": [{"shape": ["batch", 3, 64, 64]}]},
    {"input": [{"shape": ["batch", 3, 128, 128]}]}
]

HISTOGRAMS = [
    ({"Add": 10, "Mul": 4}, {"Add": 8, "Mul": 4}),
    ({"Conv": 5, "Relu": 5}, {"Conv": 3, "Relu": 6})
]

def unbake_concat(graph_dict):
    nodes = graph_dict.get("node", [])
    fixed_nodes = []
    for node in nodes:
        if node.get("op") == "Concat":
            attrs = node.get("attribute", [])
            new_attrs = []
            for attr in attrs:
                if attr.get("name") == "axis" and attr.get("i") == 0:
                    attr = dict(attr)
                    attr["_unbaked_batch_one"] = True
                new_attrs.append(attr)
            node = dict(node)
            node["attribute"] = new_attrs
        fixed_nodes.append(node)
    return {"node": fixed_nodes, "unbaked": True}

def assert_symbolic_axes(onnx_model, batch_dim_name="batch"):
    inputs = onnx_model.get("input", [])
    for inp in inputs:
        shape = inp.get("shape", [])
        if shape and shape[0] != batch_dim_name:
            raise AssertionError(f"Expected dynamic axis {batch_dim_name}, got {shape[0]}")
    return True

def op_histogram_diff(dynamo_ops, ts_ops):
    all_keys = set(dynamo_ops.keys()) | set(ts_ops.keys())
    diff = {}
    for k in all_keys:
        diff[k] = abs(dynamo_ops.get(k, 0) - ts_ops.get(k, 0))
    return diff
