def tag_ops(op_names):
    fusible = {"add", "mul", "relu", "sigmoid", "broadcast"}
    return [op in fusible for op in op_names]
