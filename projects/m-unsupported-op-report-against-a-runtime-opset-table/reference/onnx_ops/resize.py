def compute_resize_shape(input_shape, scales, sizes, mode):
    if sizes is not None and len(sizes) > 0:
        return [int(s) for s in sizes]
    if scales is not None and len(scales) > 0:
        return [int(dim * scale) for dim, scale in zip(input_shape, scales)]
    return list(input_shape)
