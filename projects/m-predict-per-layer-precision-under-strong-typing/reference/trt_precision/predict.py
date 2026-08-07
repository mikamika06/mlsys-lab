def predict_layer_precisions(graph):
    """Predict effective per-layer execution precisions in strongly typed mode."""
    tensor_types = {}
    for inp in graph.get("inputs", []):
        tensor_types[inp["name"]] = inp["dtype"]

    layer_precisions = {}
    for node in graph.get("nodes", []):
        node_id = node["id"]
        op_type = node["op"]
        inputs = node.get("inputs", [])
        explicit_p = node.get("explicit_precision")

        if op_type == "Cast":
            target_dtype = node["target_dtype"]
            eff_p = explicit_p if explicit_p else target_dtype
            layer_precisions[node_id] = eff_p
            for out in node.get("outputs", []):
                tensor_types[out] = target_dtype
            continue

        in_dtypes = [tensor_types.get(i, "FP32") for i in inputs]

        if explicit_p:
            eff_p = explicit_p
        else:
            if len(in_dtypes) > 0 and all(d == "FP16" for d in in_dtypes):
                eff_p = "FP16"
            elif any(d == "FP32" for d in in_dtypes):
                eff_p = "FP32"
            elif any(d == "TF32" for d in in_dtypes):
                eff_p = "TF32"
            else:
                eff_p = in_dtypes[0] if in_dtypes else "FP32"

        layer_precisions[node_id] = eff_p
        for out in node.get("outputs", []):
            tensor_types[out] = eff_p

    return layer_precisions
