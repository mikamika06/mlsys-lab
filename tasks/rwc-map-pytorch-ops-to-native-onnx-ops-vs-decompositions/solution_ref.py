def map_ops(ops_list):
    oracle = {
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
    return [oracle.get(op, False) for op in ops_list]
