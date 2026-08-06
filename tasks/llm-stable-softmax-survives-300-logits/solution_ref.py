import math

def stable_softmax(logits):
    """Compute softmax along the last axis, numerically stable.

    Subtracts the per-row maximum before exponentiating to avoid overflow.
    """
    def _get_shape(lst):
        shape = []
        curr = lst
        while isinstance(curr, list):
            shape.append(len(curr))
            if len(curr) == 0:
                break
            curr = curr[0]
        return tuple(shape)

    def _flatten(lst):
        if not isinstance(lst, list):
            return [float(lst)]
        res = []
        for item in lst:
            res.extend(_flatten(item))
        return res

    def _unflatten(flat, shape):
        if len(shape) == 1:
            return [float(x) for x in flat[:shape[0]]]
        sub_size = 1
        for s in shape[1:]:
            sub_size *= s
        res = []
        for i in range(shape[0]):
            res.append(_unflatten(flat[i * sub_size : (i + 1) * sub_size], shape[1:]))
        return res

    if not isinstance(logits, list):
        logits = list(logits)

    shape = _get_shape(logits)
    if not shape:
        return []

    K = shape[-1]
    flat = _flatten(logits)
    N = len(flat) // K

    out_flat = []
    for i in range(N):
        row = flat[i * K : (i + 1) * K]
        m = row[0]
        for j in range(1, K):
            val = row[j]
            if val > m:
                m = val
        sum_e = 0.0
        e_row = [0.0] * K
        for j in range(K):
            val = math.exp(row[j] - m)
            e_row[j] = val
            sum_e += val
        for j in range(K):
            out_flat.append(e_row[j] / sum_e)

    return _unflatten(out_flat, shape)
