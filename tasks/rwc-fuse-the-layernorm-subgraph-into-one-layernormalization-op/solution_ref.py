import math


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


def _to_python(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_python(item) for item in obj]
    return obj


def _apply_layernorm_1d(x_1d, gamma, beta, eps):
    d = len(x_1d)
    acc = 0.0
    for j in range(d):
        acc += float(x_1d[j])
    mean_val = acc / d

    acc_var = 0.0
    for j in range(d):
        diff = float(x_1d[j]) - mean_val
        acc_var += diff * diff
    var_val = acc_var / d

    sqrt_val = math.sqrt(var_val + eps)

    out_1d = []
    for j in range(d):
        g = float(gamma[j])
        b = float(beta[j])
        val = float(x_1d[j])
        out_1d.append(g * (val - mean_val) / sqrt_val + b)
    return out_1d


def _process_nested(x, gamma, beta, eps):
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [_process_nested(sub, gamma, beta, eps) for sub in x]
    return _apply_layernorm_1d(x, gamma, beta, eps)


def fuse_layernorm_subgraph(nodes: list[dict], inputs: dict) -> dict:
    x = _to_python(inputs["x"])
    gamma = _to_python(inputs["gamma"])
    beta = _to_python(inputs["beta"])
    eps = float(inputs.get("epsilon", 1e-5))

    output = _process_nested(x, gamma, beta, eps)

    return {
        "fused_span": _detect_span(nodes),
        "output": output,
    }
