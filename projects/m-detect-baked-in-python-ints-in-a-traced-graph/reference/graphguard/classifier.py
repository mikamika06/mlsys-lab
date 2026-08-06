def classify_failures(messages):
    """Classify export failures."""
    mapping = {
        "0 elements": "dynamic_shape",
        "dimension 0": "dynamic_shape",
        "custom op": "unsupported_op",
        "python data type": "unsupported_op",
        "static int": "baked_int",
        "integer constant": "baked_int",
        "scalar literal": "baked_int",
        "non-tensor boolean": "control_flow",
        "depends on intermediate": "control_flow",
        "higher-order control flow": "control_flow",
        "side effect": "side_effect",
        "mutating tensor storage": "side_effect"
    }
    res = []
    for msg in messages:
        cat = "unknown"
        for k, v in mapping.items():
            if k in msg:
                cat = v
                break
        res.append(cat)
    return res
