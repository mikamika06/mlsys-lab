import numpy as np


def minimal_op_config(model_set):
    ops = set()
    for model in model_set:
        for node in model.get("nodes", []):
            if node.get("provider") == "CPU":
                ops.add(node.get("op"))
    return sorted(list(ops))
