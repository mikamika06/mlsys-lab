import math
import itertools


def _get_shape(lst):
    shape = []
    curr = lst
    while isinstance(curr, list):
        shape.append(len(curr))
        if len(curr) > 0:
            curr = curr[0]
        else:
            break
    return tuple(shape)


def _get_item(lst, idx):
    curr = lst
    for i in idx:
        curr = curr[i]
    return curr


def _set_item(lst, idx, val):
    curr = lst
    for i in idx[:-1]:
        curr = curr[i]
    curr[idx[-1]] = val


def _empty_like(lst):
    shape = _get_shape(lst)
    def make_nested(dims):
        if not dims:
            return 0.0
        return [make_nested(dims[1:]) for _ in range(dims[0])]
    return make_nested(shape)


def _log_softmax(x):
    shape = _get_shape(x)
    out = _empty_like(x)

    if len(shape) == 0:
        return 0.0

    prefix_shape = shape[:-1]
    last_dim = shape[-1]

    if not prefix_shape:
        m = x[0]
        for i in range(1, last_dim):
            if x[i] > m:
                m = x[i]

        exp_sum = 0.0
        for i in range(last_dim):
            exp_sum += math.exp(x[i] - m)

        log_exp_sum = math.log(exp_sum)
        for i in range(last_dim):
            out[i] = (x[i] - m) - log_exp_sum
    else:
        for p in itertools.product(*(range(d) for d in prefix_shape)):
            m = _get_item(x, p + (0,))
            for i in range(1, last_dim):
                val = _get_item(x, p + (i,))
                if val > m:
                    m = val

            exp_sum = 0.0
            for i in range(last_dim):
                exp_sum += math.exp(_get_item(x, p + (i,)) - m)

            log_exp_sum = math.log(exp_sum)
            for i in range(last_dim):
                _set_item(out, p + (i,), (_get_item(x, p + (i,)) - m) - log_exp_sum)

    return out


def log_softmax_vjp(x: list[list[float]], g: list[list[float]]) -> list[list[float]]:
    """Vector-Jacobian product of y = log_softmax(x, axis=-1).

    Given the upstream gradient `g` (dLoss/dy, same shape as x), returns
    dLoss/dx = g - softmax(x) * sum(g, axis=-1, keepdims=True).
    """
    shape = _get_shape(x)
    out = _empty_like(x)

    log_s = _log_softmax(x)

    if len(shape) == 0:
        return g - math.exp(log_s) * g

    prefix_shape = shape[:-1]
    last_dim = shape[-1]

    if not prefix_shape:
        softmax_arr = [math.exp(log_s[i]) for i in range(last_dim)]
        g_sum = 0.0
        for i in range(last_dim):
            g_sum += g[i]

        for i in range(last_dim):
            out[i] = g[i] - softmax_arr[i] * g_sum
    else:
        for p in itertools.product(*(range(d) for d in prefix_shape)):
            g_sum = 0.0
            for i in range(last_dim):
                g_sum += _get_item(g, p + (i,))

            for i in range(last_dim):
                sm = math.exp(_get_item(log_s, p + (i,)))
                res = _get_item(g, p + (i,)) - sm * g_sum
                _set_item(out, p + (i,), res)

    return out
