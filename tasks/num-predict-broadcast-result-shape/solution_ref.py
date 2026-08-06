def broadcast_shape(shape1, shape2):
    """
    Return the broadcast result shape of two input shapes.
    If the shapes are not compatible, return an empty tuple ().
    """
    res = []
    i = len(shape1) - 1
    j = len(shape2) - 1
    while i >= 0 or j >= 0:
        dim1 = shape1[i] if i >= 0 else 1
        dim2 = shape2[j] if j >= 0 else 1
        if dim1 == dim2:
            res.append(dim1)
        elif dim1 == 1:
            res.append(dim2)
        elif dim2 == 1:
            res.append(dim1)
        else:
            return ()
        i -= 1
        j -= 1
    return tuple(reversed(res))
