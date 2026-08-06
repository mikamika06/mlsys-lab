def quantize_fp8(x, scale, ebits=4, mbits=3):
    raise NotImplementedError


def dequantize_fp8(x_q, scale):
    raise NotImplementedError


def compute_rel_error(x, x_rec):
    raise NotImplementedError
