def broadcast_shape(shape_a, shape_b):
    """Return the broadcast result shape or 'incompatible'."""
    a = tuple(shape_a)
    b = tuple(shape_b)

    if len(a) > len(b):
        b = (1,) * (len(a) - len(b)) + b
    elif len(b) > len(a):
        a = (1,) * (len(b) - len(a)) + a

    result = []
    for x, y in zip(a, b):
        if x == y:
            result.append(x)
        elif x == 1:
            result.append(y)
        elif y == 1:
            result.append(x)
        else:
            return "incompatible"
    return tuple(result)
