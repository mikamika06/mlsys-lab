def compute_scale(tensor, qmin, qmax):
    raise NotImplementedError


def quantize(tensor, scale, qmin, qmax):
    raise NotImplementedError


def dequantize(codes, scale):
    raise NotImplementedError
