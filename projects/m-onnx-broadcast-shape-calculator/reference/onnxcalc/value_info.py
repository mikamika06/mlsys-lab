from onnxcalc.broadcast import compute_broadcast_shape


def infer_graph_value_info(graph):
    info = {}
    for inp in graph.get("inputs", []):
        info[inp["name"]] = {
            "shape": list(inp["shape"]),
            "type": inp.get("type", "float32")
        }

    existing = graph.get("value_info", {})
    if isinstance(existing, dict):
        for k, v in existing.items():
            info[k] = {"shape": list(v["shape"]), "type": v.get("type", "float32")}
    elif isinstance(existing, list):
        for item in existing:
            info[item["name"]] = {"shape": list(item["shape"]), "type": item.get("type", "float32")}

    for node in graph.get("nodes", []):
        op = node["op"]
        in_names = node["inputs"]
        out_names = node["outputs"]

        for name in in_names:
            if name not in info:
                raise ValueError(f"Missing shape for tensor: {name}")

        if op in ("Add", "Sub", "Mul", "Div"):
            if len(in_names) != 2:
                raise ValueError(f"Op {op} requires 2 inputs")
            s1 = info[in_names[0]]["shape"]
            s2 = info[in_names[1]]["shape"]
            out_shape = compute_broadcast_shape(s1, s2)
            out_type = info[in_names[0]]["type"]
        elif op in ("Relu", "Sigmoid", "Abs"):
            out_shape = list(info[in_names[0]]["shape"])
            out_type = info[in_names[0]]["type"]
        elif op == "Reshape":
            attrs = node.get("attrs", {})
            out_shape = list(attrs.get("shape", []))
            out_type = info[in_names[0]]["type"]
        elif op == "MatMul":
            if len(in_names) != 2:
                raise ValueError("MatMul requires 2 inputs")
            s1 = info[in_names[0]]["shape"]
            s2 = info[in_names[1]]["shape"]
            if len(s1) < 2 or len(s2) < 2:
                raise ValueError("MatMul inputs must have rank >= 2")
            m, k1 = s1[-2], s1[-1]
            k2, n = s2[-2], s2[-1]
            if k1 != k2 and k1 != 1 and k2 != 1 and not (isinstance(k1, str) or isinstance(k2, str)):
                raise ValueError(f"MatMul inner dimension mismatch: {k1} vs {k2}")
            batch_shape = compute_broadcast_shape(s1[:-2], s2[:-2])
            out_shape = batch_shape + [m, n]
            out_type = info[in_names[0]]["type"]
        else:
            raise ValueError(f"Unsupported operation: {op}")

        for out_name in out_names:
            info[out_name] = {"shape": out_shape, "type": out_type}

    return info
