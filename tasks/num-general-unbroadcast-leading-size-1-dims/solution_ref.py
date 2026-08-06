def unbroadcast(grad, shape):
    """Reduce `grad` (which has the shape produced by broadcasting a tensor
    of shape `shape` up to grad's shape) back down to `shape`, by summing over
    every axis that broadcasting introduced or stretched.

    Two independent kinds of axes have to be summed away:
      1. Extra LEADING axes that `shape` doesn't have at all (broadcasting inserts
         these on the left when aligning ranks).
      2. Axes where `shape` itself is 1 but `grad` is wider (broadcasting stretched
         a size-1 axis).

    Returns a nested list of floats (or float for scalar) with exactly
    `len(shape)` dimensions, equal to `shape`.
    """
    shape = tuple(shape)

    def _get_shape_and_flatten(obj):
        if not isinstance(obj, list):
            return (), [float(obj)]
        shp = []
        curr = obj
        while isinstance(curr, list):
            shp.append(len(curr))
            if len(curr) > 0:
                curr = curr[0]
            else:
                break

        elems = []
        def _flatten(item):
            if isinstance(item, list):
                for x in item:
                    _flatten(x)
            else:
                elems.append(float(item))

        _flatten(obj)
        return tuple(shp), elems

    def _build_nested(shp, flat_data):
        if not shp:
            return flat_data[0]
        idx = 0

        def _build(dim_idx):
            nonlocal idx
            if dim_idx == len(shp) - 1:
                res = flat_data[idx:idx + shp[dim_idx]]
                idx += shp[dim_idx]
                return res
            res = []
            for _ in range(shp[dim_idx]):
                res.append(_build(dim_idx + 1))
            return res

        return _build(0)

    g_shape, flat_grad = _get_shape_and_flatten(grad)
    g_size = len(flat_grad)
    g_rank = len(g_shape)

    g_strides = []
    acc = 1
    for dim in reversed(g_shape):
        g_strides.append(acc)
        acc *= dim
    g_strides.reverse()

    target_shape = (1,) * (g_rank - len(shape)) + shape

    target_strides = []
    acc = 1
    for dim in reversed(target_shape):
        target_strides.append(acc)
        acc *= dim
    target_strides.reverse()

    out_size = 1
    for dim in shape:
        out_size *= dim

    flat_out = [0.0] * out_size

    for idx in range(g_size):
        rem = idx
        out_idx = 0
        for r in range(g_rank):
            coord = rem // g_strides[r]
            rem %= g_strides[r]
            if target_shape[r] != 1:
                out_idx += coord * target_strides[r]

        flat_out[out_idx] += flat_grad[idx]

    return _build_nested(shape, flat_out)
