def _oracle() -> dict:
    return {
        "add": True,
        "mul": True,
        "conv2d": False,
        "relu": True,
        "softmax": True,
        "batch_norm": False,
        "layer_norm": False,
        "matmul": True,
        "transpose": True,
        "reshape": True,
        "mean": True,
        "sum": True,
        "log_softmax": True,
        "sigmoid": True,
        "tanh": True,
        "leaky_relu": True,
        "dropout": False
    }

def grade(sol, fx) -> dict:
    oracle = _oracle()
    cases = [
        ["add", "mul", "conv2d"],
        ["relu", "softmax", "batch_norm"],
        ["matmul", "transpose", "reshape"],
        ["mean", "sum", "log_softmax"],
        ["sigmoid", "tanh", "leaky_relu", "dropout"]
    ]
    ok = 1.0
    for ops in cases:
        try:
            got = sol.map_ops(ops)
        except Exception:
            return {"exact_match": 0.0}
        if len(got) != len(ops):
            return {"exact_match": 0.0}
        for op, val in zip(ops, got):
            expected = oracle.get(op, False)
            if val != expected:
                return {"exact_match": 0.0}
    return {"exact_match": ok}
