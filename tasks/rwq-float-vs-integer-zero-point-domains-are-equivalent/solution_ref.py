def dual_zero_point_dequant(codes: list[int], scale: float, zp_float: float):
    zp_int = -zp_float / scale
    deq_float_domain = [scale * float(q) + zp_float for q in codes]
    deq_int_domain = [scale * (float(q) - zp_int) for q in codes]
    return deq_float_domain, deq_int_domain, float(zp_int)
