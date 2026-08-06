import numpy as np


def fuse_ops(x, y, z):
    return (x + y) * z


def hbm_bytes_saved(shapes):
    total_unfused = 0
    total_fused = 0
    for x_shape, y_shape, z_shape, out_shape in shapes:
        x_numel = np.prod(x_shape)
        y_numel = np.prod(y_shape)
        z_numel = np.prod(z_shape)
        out_numel = np.prod(out_shape)
        elem_size = 4
        unfused = (x_numel + y_numel + out_numel + y_numel + z_numel + out_numel) * elem_size
        fused = (x_numel + y_numel + z_numel + out_numel) * elem_size
        total_unfused += unfused
        total_fused += fused
    return int(total_unfused - total_fused)
