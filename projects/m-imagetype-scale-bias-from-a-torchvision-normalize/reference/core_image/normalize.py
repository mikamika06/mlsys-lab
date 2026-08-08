def compute_scale_bias(mean, std):
    scale = [1.0 / (255.0 * s) for s in std]
    bias = [-m / s for m, s in zip(mean, std)]
    return scale, bias
