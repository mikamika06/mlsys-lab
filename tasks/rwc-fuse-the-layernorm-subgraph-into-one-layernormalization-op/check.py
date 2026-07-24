import numpy as np


def _detect_layernorm_span(nodes):
    by_name = {n["name"]: n for n in nodes}
    ops = [n["op"] for n in nodes]
    target = [
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
    if ops != target:
        return []
    for a, b in zip(nodes, nodes[1:]):
        if b["inputs"] and a["name"] not in b["inputs"]:
            if b["op"] not in ("Sub", "Div", "Mul", "Add"):
                return []
    return [n["name"] for n in nodes]


def _layernorm_oracle(x, gamma, beta, eps):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def grade(sol, fx) -> dict:
    nodes = [
        {"name": "mean", "op": "ReduceMean", "inputs": ["x"]},
        {"name": "sub", "op": "Sub", "inputs": ["x", "mean"]},
        {"name": "pow", "op": "Pow", "inputs": ["sub"]},
        {"name": "var", "op": "ReduceMean", "inputs": ["pow"]},
        {"name": "eps", "op": "Add", "inputs": ["var"]},
        {"name": "sqrt", "op": "Sqrt", "inputs": ["eps"]},
        {"name": "div", "op": "Div", "inputs": ["sub", "sqrt"]},
        {"name": "scale", "op": "Mul", "inputs": ["div", "gamma"]},
        {"name": "bias", "op": "Add", "inputs": ["scale", "beta"]},
    ]

    x = np.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, -2.0, 0.5]],
        dtype=np.float32,
    )
    gamma = np.array([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
    beta = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    eps = 1e-5

    expected_span = _detect_layernorm_span(nodes)
    expected = _layernorm_oracle(x, gamma, beta, eps)

    try:
        got = sol.fuse_layernorm_subgraph(
            nodes,
            {
                "x": x,
                "gamma": gamma,
                "beta": beta,
                "epsilon": eps,
            },
        )
        err = float(np.max(np.abs(np.asarray(got["output"], dtype=np.float64) - expected)))
        span_ok = 1.0 if list(got["fused_span"]) == expected_span else 0.0
    except Exception:
        err = float("inf")
        span_ok = 0.0

    return {
        "max_abs_err": 0.0 if err <= 1e-6 else 1.0,
        "span_match": span_ok,
    }
