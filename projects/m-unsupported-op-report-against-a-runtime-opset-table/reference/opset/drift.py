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
