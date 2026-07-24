def compute_8da4w_efficiency(weight_shape, group_size):
    """
    Compute the effective bits per parameter and size ratio for an 8da4w linear layer.

    Parameters
    ----------
    weight_shape : tuple[int, int]
        Shape of the full‑precision weight matrix (out_features, in_features).
    group_size : int
        Number of input columns that share a single FP16 scale.

    Returns
    -------
    tuple[float, float]
        (bits_per_param, size_ratio)
    """
    out_features, in_features = weight_shape
    num_weights = out_features * in_features

    # Packed int4 weights: two per byte
    weight_bytes = (num_weights + 1) // 2

    # FP16 scales per group of columns
    groups = (in_features + group_size - 1) // group_size
    scale_bytes = groups * 2

    total_bytes = weight_bytes + scale_bytes
    total_bits = total_bytes * 8

    bits_per_param = total_bits / num_weights
    size_ratio = (num_weights * 2) / total_bytes

    return float(bits_per_param), float(size_ratio)
