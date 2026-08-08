def prune_then_quantize(W, mask, scale, zero_point):
    raise NotImplementedError


def quantize_then_prune(W, mask, scale, zero_point):
    raise NotImplementedError


def evaluate_accuracy(W_orig, W_compressed):
    raise NotImplementedError
