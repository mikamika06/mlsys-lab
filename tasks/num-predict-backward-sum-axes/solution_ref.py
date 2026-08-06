def predict_backward_sum_axes(a_shape, b_shape):
    max_len = max(len(a_shape), len(b_shape))
    a_padded = (1,) * (max_len - len(a_shape)) + tuple(a_shape)
    b_padded = (1,) * (max_len - len(b_shape)) + tuple(b_shape)

    c_shape = []
    for dim_a, dim_b in zip(a_padded, b_padded):
        if dim_a == 1:
            c_shape.append(dim_b)
        elif dim_b == 1:
            c_shape.append(dim_a)
        elif dim_a == dim_b:
            c_shape.append(dim_a)
        else:
            return ((0,), (0,))

    a_offset = max_len - len(a_shape)
    a_axes = tuple(i for i, s in enumerate(a_shape) if s != 1 and c_shape[i + a_offset] != s)

    b_offset = max_len - len(b_shape)
    b_axes = tuple(i for i, s in enumerate(b_shape) if s != 1 and c_shape[i + b_offset] != s)

    return (a_axes, b_axes)
