import numpy as np


def consolidate(shards, mapping):
    result = {}
    for target_key, spec in mapping.items():
        src_keys = spec["sources"]
        op = spec.get("op", "stack")
        axis = spec.get("axis", 0)
        tensors = [shards[s_key] for s_key in src_keys]
        if op == "stack":
            result[target_key] = np.stack(tensors, axis=axis)
        elif op == "concat":
            result[target_key] = np.concatenate(tensors, axis=axis)
        elif op == "copy":
            result[target_key] = tensors[0].copy()
        else:
            raise ValueError(f"unknown op {op}")
    return result
