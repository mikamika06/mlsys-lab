def sum_to_shape(grad: list, input_shape: tuple) -> list:
    """Reduce grad by summing along broadcasted dimensions to produce input_shape."""
    def get_shape(x):
        if not isinstance(x, list):
            return ()
        if len(x) == 0:
            return (0,)
        return (len(x),) + get_shape(x[0])

    def get_item(x, idx):
        curr = x
        for i in idx:
            curr = curr[i]
        return curr

    def set_item(x, idx, val):
        curr = x
        for i in idx[:-1]:
            curr = curr[i]
        curr[idx[-1]] = val

    def zeros(shape):
        if not shape:
            return 0.0
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [zeros(shape[1:]) for _ in range(shape[0])]

    target_shape = get_shape(grad)
    if not target_shape:
        return grad

    ndim_diff = len(target_shape) - len(input_shape)
    padded = (1,) * ndim_diff + tuple(input_shape)
    axes_to_sum = tuple(
        i for i in range(len(target_shape))
        if padded[i] == 1 and target_shape[i] > 1
    )
    reduced_shape = tuple(
        1 if i in axes_to_sum else target_shape[i]
        for i in range(len(target_shape))
    )

    if axes_to_sum:
        out = zeros(reduced_shape)
        sum_shapes = tuple(target_shape[i] for i in axes_to_sum)

        def iter_indices(shape):
            if not shape:
                yield ()
                return
            if len(shape) == 1:
                for i in range(shape[0]):
                    yield (i,)
                return
            for i in range(shape[0]):
                for sub in iter_indices(shape[1:]):
                    yield (i,) + sub

        for out_idx in iter_indices(reduced_shape):
            acc = 0.0
            for sub_idx in iter_indices(sum_shapes):
                target_idx = list(out_idx)
                for ax, s_idx in zip(axes_to_sum, sub_idx):
                    target_idx[ax] = s_idx
                acc += get_item(grad, tuple(target_idx))
            if reduced_shape == ():
                out = acc
            else:
                set_item(out, out_idx, acc)
    else:
        out = grad

    def flatten(x):
        res = []
        def _flat(sub):
            if not isinstance(sub, list):
                res.append(sub)
            else:
                for item in sub:
                    _flat(item)
        _flat(x)
        return res

    flat_data = flatten(out)

    def unflatten(flat_iter, shape):
        if not shape:
            return next(flat_iter)
        if len(shape) == 1:
            return [next(flat_iter) for _ in range(shape[0])]
        return [unflatten(flat_iter, shape[1:]) for _ in range(shape[0])]

    return unflatten(iter(flat_data), input_shape)
