import numpy as np
from numpy.lib.stride_tricks import as_strided


def broadcast_to_strided(a, shape):
    a = np.asarray(a)
    shape = tuple(int(s) for s in shape)

    if len(shape) < a.ndim:
        raise ValueError(
            f"cannot broadcast shape {a.shape} to fewer dimensions {shape}"
        )

    pad = len(shape) - a.ndim
    src_shape = (1,) * pad + a.shape
    src_strides = (0,) * pad + a.strides

    out_strides = []
    for tgt, dim, st in zip(shape, src_shape, src_strides):
        if dim == tgt:
            out_strides.append(0 if dim == 1 else st)
        elif dim == 1:
            out_strides.append(0)
        else:
            raise ValueError(
                f"operands could not be broadcast together: {a.shape} -> {shape}"
            )

    return as_strided(a, shape=shape, strides=tuple(out_strides), writeable=False)
