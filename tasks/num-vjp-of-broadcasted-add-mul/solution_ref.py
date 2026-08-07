"""Reference solution for `num-vjp-of-broadcasted-add-mul`."""
from __future__ import annotations


def _get_shape(nested):
    shape = []
    curr = nested
    while isinstance(curr, list):
        shape.append(len(curr))
        curr = curr[0] if len(curr) > 0 else []
    return tuple(shape)


def _zeros_like_shape(shape):
    if not shape:
        return 0.0
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    return [_zeros_like_shape(shape[1:]) for _ in range(shape[0])]


def _get_element(nested, idx):
    curr = nested
    for i in idx:
        curr = curr[i]
    return curr


def _set_element(nested, idx, val):
    curr = nested
    for i in idx[:-1]:
        curr = curr[i]
    curr[idx[-1]] = val


def _iter_indices(shape):
    if not shape:
        yield ()
        return

    def helper(dim):
        if dim == len(shape):
            yield ()
            return
        for i in range(shape[dim]):
            for rest in helper(dim + 1):
                yield (i,) + rest

    yield from helper(0)


def _sum_axis_0(arr: list) -> list:
    shape = _get_shape(arr)
    new_shape = shape[1:]
    out = _zeros_like_shape(new_shape)
    for idx in _iter_indices(new_shape):
        s = 0.0
        for d in range(shape[0]):
            s += _get_element(arr, (d,) + idx)
        if new_shape:
            _set_element(out, idx, s)
        else:
            out = s
    return out


def _sum_axis_i_keepdims(arr: list, i: int) -> list:
    shape = _get_shape(arr)
    new_shape = list(shape)
    new_shape[i] = 1
    new_shape = tuple(new_shape)
    out = _zeros_like_shape(new_shape)
    for idx in _iter_indices(new_shape):
        s = 0.0
        idx_list = list(idx)
        for d in range(shape[i]):
            idx_list[i] = d
            s += _get_element(arr, tuple(idx_list))
        _set_element(out, idx, s)
    return out


def _broadcast_shapes(s1, s2):
    diff = len(s1) - len(s2)
    if diff > 0:
        s2 = (1,) * diff + s2
    elif diff < 0:
        s1 = (1,) * (-diff) + s1

    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        elif d1 == d2:
            out_shape.append(d1)
        else:
            raise ValueError(f"Incompatible shapes {s1} and {s2}")
    return tuple(out_shape)


def _broadcast_get(nested, shape, target_shape):
    diff = len(target_shape) - len(shape)
    full_shape = (1,) * diff + shape

    out = _zeros_like_shape(target_shape)
    for idx in _iter_indices(target_shape):
        src_idx = []
        offset = len(target_shape) - len(shape)
        for i, idx_val in enumerate(idx):
            if i < offset:
                src_idx.append(0)
            else:
                orig_dim = shape[i - offset]
                if orig_dim == 1:
                    src_idx.append(0)
                else:
                    src_idx.append(idx_val)
        val = _get_element(nested, tuple(src_idx))
        _set_element(out, idx, val)
    return out


def _sum_to_shape(grad: list, shape: tuple) -> list:
    grad_shape = _get_shape(grad)
    extra = len(grad_shape) - len(shape)
    for _ in range(extra):
        grad = _sum_axis_0(grad)
        grad_shape = _get_shape(grad)

    for i, s in enumerate(shape):
        if s == 1 and grad_shape[i] != 1:
            grad = _sum_axis_i_keepdims(grad, i)
            grad_shape = _get_shape(grad)

    if grad_shape == shape:
        return grad

    def reshape_nested(val, src_shape, dst_shape):
        if src_shape == dst_shape:
            return val
        flat = []
        def flatten(v, s):
            if not s:
                flat.append(v)
                return
            if len(s) == 1:
                flat.extend(v)
                return
            for sub in v:
                flatten(sub, s[1:])
        flatten(val, src_shape)

        def unflatten(s):
            if not s:
                return flat.pop(0)
            if len(s) == 1:
                return [flat.pop(0) for _ in range(s[0])]
            return [unflatten(s[1:]) for _ in range(s[0])]

        return unflatten(dst_shape)

    return reshape_nested(grad, grad_shape, shape)


def add_vjp(a: list, b: list, grad_out: list):
    """VJP of `out = a + b`: returns (grad_a, grad_b)."""
    a_shape = _get_shape(a)
    b_shape = _get_shape(b)

    grad_a = _sum_to_shape(grad_out, a_shape)
    grad_b = _sum_to_shape(grad_out, b_shape)
    return grad_a, grad_b


def mul_vjp(a: list, b: list, grad_out: list):
    """VJP of `out = a * b`: returns (grad_a, grad_b)."""
    a_shape = _get_shape(a)
    b_shape = _get_shape(b)

    b_broadcasted = _broadcast_get(b, a_shape, _broadcast_shapes(a_shape, b_shape))
    a_broadcasted = _broadcast_get(a, b_shape, _broadcast_shapes(a_shape, b_shape))

    def elementwise_mul(v1, v2, shape):
        out = _zeros_like_shape(shape)
        for idx in _iter_indices(shape):
            val1 = _get_element(v1, idx)
            val2 = _get_element(v2, idx)
            _set_element(out, idx, val1 * val2)
        return out

    out_shape = _get_shape(grad_out)
    prod_a = elementwise_mul(grad_out, b_broadcasted, out_shape)
    prod_b = elementwise_mul(grad_out, a_broadcasted, out_shape)

    grad_a = _sum_to_shape(prod_a, a_shape)
    grad_b = _sum_to_shape(prod_b, b_shape)
    return grad_a, grad_b
