import numpy as np


def _oracle(nodes):
    values = {}
    constant = {}
    folded = 0
    for node in nodes:
        op = node["op"]
        name = node["name"]
        if op == "const":
            values[name] = np.asarray(node["value"], dtype=np.float64)
            constant[name] = True
        elif op == "mul":
            a, b = node["inputs"]
            values[name] = values[a] * values[b]
            constant[name] = constant[a] and constant[b]
            if constant[name]:
                folded += 1
        elif op == "add":
            a, b = node["inputs"]
            values[name] = values[a] + values[b]
            constant[name] = constant[a] and constant[b]
            if constant[name]:
                folded += 1
        elif op == "identity":
            src = node["inputs"][0]
            values[name] = values[src]
            constant[name] = constant[src]
            if constant[name]:
                folded += 1
    return np.asarray(values[nodes[-1]["name"]], dtype=np.float64), folded


def grade(sol, fx) -> dict:
    cases = [
        [
            {"name": "w", "op": "const", "value": np.array([2.0, -1.0])},
            {"name": "s", "op": "const", "value": np.array([3.0, 4.0])},
            {"name": "m", "op": "mul", "inputs": ["w", "s"]},
            {"name": "b", "op": "const", "value": np.array([1.0, 2.0])},
            {"name": "y", "op": "add", "inputs": ["m", "b"]},
        ],
        [
            {"name": "a", "op": "const", "value": np.array([[1.0, 2.0], [3.0, 4.0]])},
            {"name": "b", "op": "const", "value": np.array([[5.0, 6.0], [7.0, 8.0]])},
            {"name": "m", "op": "mul", "inputs": ["a", "b"]},
            {"name": "o", "op": "identity", "inputs": ["m"]},
        ],
        [
            {"name": "x", "op": "const", "value": np.array([10.0, 20.0, 30.0])},
            {"name": "scale", "op": "const", "value": np.array([0.5, 2.0, -1.0])},
            {"name": "mul", "op": "mul", "inputs": ["x", "scale"]},
            {"name": "bias", "op": "const", "value": np.array([1.0, 1.0, 1.0])},
            {"name": "out", "op": "add", "inputs": ["mul", "bias"]},
            {"name": "final", "op": "identity", "inputs": ["out"]},
        ],
    ]

    max_err = 0.0
    count_ok = 1.0

    for nodes in cases:
        ref_tensor, ref_count = _oracle(nodes)
        try:
            got_tensor, got_count = sol.fold_constants(nodes)
            got_tensor = np.asarray(got_tensor, dtype=np.float64)
            err = float(np.max(np.abs(got_tensor - ref_tensor)))
            max_err = max(max_err, err)
            if int(got_count) != int(ref_count):
                count_ok = 0.0
        except Exception:
            return {"max_abs_err": float("inf"), "folded_nodes": 0.0}

    return {"max_abs_err": max_err, "folded_nodes": count_ok}
