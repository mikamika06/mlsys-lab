def check_support(nodes, model_opset, ops_table):
    unsupported = []
    for n in nodes:
        op = n["op_type"]
        if op not in ops_table:
            unsupported.append(n["name"])
        elif model_opset > ops_table[op]:
            unsupported.append(n["name"])
    return unsupported


def migrate_squeeze_11(nodes):
    out = []
    for n in nodes:
        if n["op_type"] == "Squeeze" and "axes" in n.get("attributes", {}):
            axes = n["attributes"]["axes"]
            new_attrs = {k: v for k, v in n["attributes"].items() if k != "axes"}
            const_name = n["name"] + "_axes"
            out.append({
                "name": const_name,
                "op_type": "Constant",
                "inputs": [],
                "outputs": [const_name + "_out"],
                "attributes": {"value": axes}
            })
            new_n = dict(n)
            new_n["attributes"] = new_attrs
            new_n["inputs"] = list(n["inputs"]) + [const_name + "_out"]
            out.append(new_n)
        else:
            out.append(dict(n))
    return out


def infer_resize_shape(input_shape, scales, sizes, opset):
    if opset >= 11 and sizes is not None and len(sizes) > 0:
        return list(sizes)
    return [int(d * s) for d, s in zip(input_shape, scales)]


NODES_1 = [
    {"name": "n1", "op_type": "Conv", "inputs": ["X"], "outputs": ["Y"], "attributes": {}},
    {"name": "n2", "op_type": "Squeeze", "inputs": ["Y"], "outputs": ["Z"], "attributes": {"axes": [0, 2]}},
    {"name": "n3", "op_type": "Resize", "inputs": ["Z"], "outputs": ["W"], "attributes": {}}
]
TABLE_1 = {"Conv": 14, "Squeeze": 10}
OPSET_1 = 11

NODES_2 = [
    {"name": "n1", "op_type": "Squeeze", "inputs": ["A"], "outputs": ["B"], "attributes": {"axes": [1]}}
]
