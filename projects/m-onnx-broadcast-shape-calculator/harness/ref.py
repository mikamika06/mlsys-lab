def compute_broadcast_shape(shape_a, shape_b):
    sa = list(shape_a)
    sb = list(shape_b)
    max_rank = max(len(sa), len(sb))
    pa = [1] * (max_rank - len(sa)) + sa
    pb = [1] * (max_rank - len(sb)) + sb
    out = []
    for da, db in zip(pa, pb):
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        elif isinstance(da, str) and isinstance(db, int):
            out.append(da)
        elif isinstance(db, str) and isinstance(da, int):
            out.append(db)
        else:
            raise ValueError(f"Incompatible dimensions for broadcast: {da} and {db}")
    return out


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


def triage_graph(graph):
    try:
        val_info = infer_graph_value_info(graph)
        return {"valid": True, "errors": [], "value_info": val_info}
    except Exception as e:
        return {"valid": False, "errors": [str(e)], "value_info": {}}


TEST_SHAPES = [
    ([1, 3, 224, 224], [3, 1, 1], [1, 3, 224, 224]),
    (["batch", 1, 64], [1, 32, 64], ["batch", 32, 64]),
    ([10], [5, 1, 10], [5, 1, 10]),
    (["N", "C", 1, 1], ["N", "C", 128, 128], ["N", "C", 128, 128]),
    ([4, 8, 16], [8, 16], [4, 8, 16]),
    ([1, 5], [2, 5], [2, 5]),
    ([3, 4], [3, 5], ValueError),
    (["A", 10], ["B", 10], ValueError),
]

TEST_GRAPHS = [
    (
        {
            "inputs": [
                {"name": "A", "shape": ["batch", 3, 32, 32], "type": "float32"},
                {"name": "B", "shape": [3, 1, 1], "type": "float32"}
            ],
            "nodes": [
                {"op": "Add", "inputs": ["A", "B"], "outputs": ["t1"]},
                {"op": "Relu", "inputs": ["t1"], "outputs": ["t2"]},
                {"op": "Reshape", "inputs": ["t2"], "outputs": ["Y"], "attrs": {"shape": ["batch", 3072]}}
            ]
        },
        {
            "A": {"shape": ["batch", 3, 32, 32], "type": "float32"},
            "B": {"shape": [3, 1, 1], "type": "float32"},
            "t1": {"shape": ["batch", 3, 32, 32], "type": "float32"},
            "t2": {"shape": ["batch", 3, 32, 32], "type": "float32"},
            "Y": {"shape": ["batch", 3072], "type": "float32"}
        }
    ),
    (
        {
            "inputs": [
                {"name": "X", "shape": ["batch", 128, 64], "type": "float32"},
                {"name": "W", "shape": [64, 256], "type": "float32"},
                {"name": "Bias", "shape": [256], "type": "float32"}
            ],
            "nodes": [
                {"op": "MatMul", "inputs": ["X", "W"], "outputs": ["h1"]},
                {"op": "Add", "inputs": ["h1", "Bias"], "outputs": ["out"]}
            ]
        },
        {
            "X": {"shape": ["batch", 128, 64], "type": "float32"},
            "W": {"shape": [64, 256], "type": "float32"},
            "Bias": {"shape": [256], "type": "float32"},
            "h1": {"shape": ["batch", 128, 256], "type": "float32"},
            "out": {"shape": ["batch", 128, 256], "type": "float32"}
        }
    ),
    (
        {
            "inputs": [
                {"name": "A", "shape": [128, 64], "type": "float32"},
                {"name": "B", "shape": [128, 32], "type": "float32"}
            ],
            "nodes": [
                {"op": "Add", "inputs": ["A", "B"], "outputs": ["out"]}
            ]
        },
        ValueError
    ),
    (
        {
            "inputs": [
                {"name": "A", "shape": [10, 10], "type": "float32"}
            ],
            "nodes": [
                {"op": "Add", "inputs": ["A", "missing_tensor"], "outputs": ["out"]}
            ]
        },
        ValueError
    )
]
