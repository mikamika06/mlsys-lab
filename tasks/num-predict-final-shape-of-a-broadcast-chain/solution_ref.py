def predict_broadcast_shape(ops):
    def _broadcast(a, b):
        na, nb = len(a), len(b)
        if na < nb:
            a = (1,) * (nb - na) + a
        elif nb < na:
            b = (1,) * (na - nb) + b
        out = []
        for x, y in zip(a, b):
            if x == y:
                out.append(x)
            elif x == 1:
                out.append(y)
            elif y == 1:
                out.append(x)
            else:
                raise ValueError(f"incompatible dims {x} and {y}")
        return tuple(out)

    shape = ops[0][1]
    for _op, s in ops[1:]:
        shape = _broadcast(shape, s)
    return shape
