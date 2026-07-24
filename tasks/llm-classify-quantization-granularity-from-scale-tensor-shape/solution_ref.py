def classify_quant_granularity(weight_shape, scale_shape):
    w = tuple(weight_shape)
    s = tuple(scale_shape)
    if len(s) == 0 or (len(s) == 1 and s[0] == 1):
        return ("per_tensor", None)
    if len(s) == 1 and s[0] == w[0]:
        return ("per_channel", None)
    if len(s) == 1:
        group_size = w[0] // s[0]
        if w[0] % s[0] == 0 and group_size > 1:
            return ("per_group", group_size)
    raise ValueError("Unsupported shape combination")
