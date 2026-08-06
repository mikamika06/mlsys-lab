import math
from collections import defaultdict

def softmax(x: list[list[float]], axis: int = -1) -> list[list[float]]:
    def get_ndim(lst):
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1
        return 1 + get_ndim(lst[0])

    ndim = get_ndim(x)
    if axis < 0:
        axis += ndim

    def get_shape(lst):
        shape = []
        curr = lst
        while isinstance(curr, list):
            shape.append(len(curr))
            curr = curr[0] if curr else []
        return shape

    shape = get_shape(x)

    def generate_indices(shape):
        if not shape:
            yield ()
            return
        for i in range(shape[0]):
            for rest in generate_indices(shape[1:]):
                yield (i,) + rest

    def get_val(lst, idx):
        curr = lst
        for i in idx:
            curr = curr[i]
        return curr

    def set_val(lst, idx, val):
        curr = lst
        for i in idx[:-1]:
            curr = curr[i]
        curr[idx[-1]] = val

    def deep_copy(lst):
        if not isinstance(lst, list):
            return lst
        return [deep_copy(item) for item in lst]

    other_axes = [i for i in range(ndim) if i != axis]

    groups = defaultdict(list)
    for idx in generate_indices(shape):
        key = tuple(idx[i] for i in other_axes)
        groups[key].append(idx)

    result = deep_copy(x)

    max_dict = {}
    for key, idxs in groups.items():
        m = max(get_val(x, idx) for idx in idxs)
        max_dict[key] = m

    sum_dict = {}
    exp_x = deep_copy(x)
    for key, idxs in groups.items():
        m = max_dict[key]
        s = 0.0
        for idx in idxs:
            val = get_val(x, idx)
            e = math.exp(val - m)
            set_val(exp_x, idx, e)
            s += e
        sum_dict[key] = s

    for key, idxs in groups.items():
        s = sum_dict[key]
        for idx in idxs:
            e = get_val(exp_x, idx)
            set_val(result, idx, e / s)

    return result
