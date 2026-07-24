import numpy as np


def _detect_span(nodes):
    expected = [
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    ]
    if [n["op"] for n in nodes] == expected:
        return [n["name"] for n in nodes]
    return []


def fuse_layernorm_subgraph(nodes, inputs):
    x = np.asarray(inputs["x"], dtype=np.float64)
    gamma = np.asarray(inputs["gamma"], dtype=np.float64)
    beta = np.asarray(inputs["beta"], dtype=np.float64)
    eps = float(inputs.get("epsilon", 1e-5))

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    output = gamma * (x - mean) / np.sqrt(var + eps) + beta

    return {
        "fused_span": _detect_span(nodes),
        "output": output,
    }
