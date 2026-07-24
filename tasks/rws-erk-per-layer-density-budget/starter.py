import numpy as np


def erk_layer_densities(shapes, global_density: float, erk_power_scale: float = 1.0):
    """
    Return a per-layer density array via Erdos-Renyi-Kernel (ERK)
    allocation: raw density proportional to
    (sum(shape)/prod(shape))**erk_power_scale, rescaled by an epsilon so
    the parameter-weighted average density equals `global_density`,
    iteratively pinning any layer whose density would exceed 1.0 to
    dense and reallocating the remaining budget. See task.md.
    """
    raise NotImplementedError('your code here')
