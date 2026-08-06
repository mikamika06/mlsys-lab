import math


def compute_grid(meta, shape):
    block_x = meta.get("BLOCK_X", 128)
    block_y = meta.get("BLOCK_Y", 1)
    ndim = len(shape)
    if ndim == 1:
        total = shape[0]
        gx = math.ceil(total / block_x)
        return (gx,)
    elif ndim == 2:
        total_x, total_y = shape[0], shape[1]
        gx = math.ceil(total_x / block_x)
        gy = math.ceil(total_y / block_y)
        return (gx, gy)
    else:
        gx = math.ceil(shape[0] / block_x)
        return (gx,)
