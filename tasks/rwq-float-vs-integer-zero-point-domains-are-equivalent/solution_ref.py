import numpy as np


def dual_zero_point_dequant(codes: np.ndarray, scale: float, zp_float: float):
    codes = np.asarray(codes, dtype=np.float64)
    zp_int = -zp_float / scale
    deq_float_domain = scale * codes + zp_float
    deq_int_domain = scale * (codes - zp_int)
    return deq_float_domain, deq_int_domain, float(zp_int)
