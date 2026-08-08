def compute_image_type_params(mean, std, scale_factor=255.0):
    scale = [1.0 / (scale_factor * s) for s in std]
    bias = [-m / s for m, s in zip(mean, std)]
    return scale, bias
