import numpy as np


def make_graph():
    W1 = np.eye(4, dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    W2 = np.ones((4, 4), dtype=np.float32) * 4.0
    b2 = np.zeros(4, dtype=np.float32)
    W3 = np.eye(4, dtype=np.float32)
    b3 = np.zeros(4, dtype=np.float32)

    return [
        {"name": "fc1", "op": "linear", "w": W1, "b": b1},
        {"name": "relu1", "op": "relu"},
        {"name": "fc2", "op": "linear", "w": W2, "b": b2},
        {"name": "exp_act", "op": "exp"},
        {"name": "norm1", "op": "normalize"},
        {"name": "fc3", "op": "linear", "w": W3, "b": b3}
    ]


def make_input():
    return np.ones((1, 4), dtype=np.float32)


def run_and_isolate(graph, x, default_dtype=np.float16):
    np.seterr(all="ignore")
    curr = x.astype(default_dtype)
    first_invalid = None

    for L in graph:
        op = L["op"]
        if op == "linear":
            w = L["w"].astype(curr.dtype)
            b = L["b"].astype(curr.dtype)
            curr = np.dot(curr, w) + b
        elif op == "relu":
            curr = np.maximum(0, curr)
        elif op == "exp":
            curr = np.exp(curr)
        elif op == "normalize":
            curr = curr / np.sum(curr, axis=-1, keepdims=True)
        elif op == "cast":
            curr = curr.astype(np.float32 if L["to"] == "float32" else np.float16)

        if first_invalid is None and not np.isfinite(curr).all():
            first_invalid = L["name"]

    return curr, first_invalid


def insert_cast_nodes(graph, start_name, end_name):
    out = []
    for L in graph:
        if L["name"] == start_name:
            out.append({"name": f"{start_name}_in_cast", "op": "cast", "to": "float32"})
        out.append(L)
        if L["name"] == end_name:
            out.append({"name": f"{end_name}_out_cast", "op": "cast", "to": "float16"})
    return out
