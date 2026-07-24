import math

def _reference(weight_shape, group_size):
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
    return bits_per_param, size_ratio


def grade(sol, fx) -> dict:
    # Test case: moderate size to exercise packing and grouping
    weight_shape = (128, 256)
    group_size = 32

    try:
        got_bits, got_ratio = sol.compute_8da4w_efficiency(weight_shape, group_size)
    except Exception:
        return {"bits_per_param_rel_err": float("inf"),
                "size_ratio_rel_err": float("inf")}

    ref_bits, ref_ratio = _reference(weight_shape, group_size)

    # Relative errors
    def rel_err(a, b):
        denom = max(abs(b), 1e-12)
        return abs(a - b) / denom

    bits_err = rel_err(got_bits, ref_bits)
    ratio_err = rel_err(got_ratio, ref_ratio)

    return {"bits_per_param_rel_err": bits_err,
            "size_ratio_rel_err": ratio_err}
