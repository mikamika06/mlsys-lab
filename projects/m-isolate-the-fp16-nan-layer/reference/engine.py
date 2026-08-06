import numpy as np


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
