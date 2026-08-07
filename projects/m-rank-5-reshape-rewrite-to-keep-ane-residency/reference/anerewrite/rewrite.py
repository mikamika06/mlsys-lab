def rewrite_shape(shape):
    if len(shape) <= 5:
        return list(shape)
    res = list(shape)
    while len(res) > 5:
        val = res.pop(0)
        res[0] *= val
    return res
