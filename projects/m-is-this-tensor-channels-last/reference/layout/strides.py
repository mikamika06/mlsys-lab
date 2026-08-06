def compute_nhwc_strides(shape):
    N, C, H, W = shape
    return (H * W * C, 1, W * C, C)


def is_channels_last(shape, strides):
    return strides == compute_nhwc_strides(shape)
