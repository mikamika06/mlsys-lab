def get_scale_bias(mean, std):
    scale = [1.0 / float(s) for s in std]
    bias = [-float(m) / float(s) for m, s in zip(mean, scale)]
    return scale, bias
