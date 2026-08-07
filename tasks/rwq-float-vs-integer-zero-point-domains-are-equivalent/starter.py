def dual_zero_point_dequant(codes: list[int], scale: float, zp_float: float):
    """Dequantize codes using both the float-domain and integer-domain zero-point
    formulas, and return the derived integer-domain zero point.

    Returns (deq_float_domain, deq_int_domain, zp_int).
    """
    raise NotImplementedError('your code here')
