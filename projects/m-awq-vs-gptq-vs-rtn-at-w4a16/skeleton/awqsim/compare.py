def quantize_rtn(W, n_bits=4, group_size=32):
    raise NotImplementedError


def quantize_gptq(W, X, n_bits=4, group_size=32):
    raise NotImplementedError


def quantize_awq(W, X, n_bits=4, group_size=32, max_scale_ratio=5.0):
    raise NotImplementedError


def compare_methods(W, X, n_bits=4, group_size=32):
    raise NotImplementedError
