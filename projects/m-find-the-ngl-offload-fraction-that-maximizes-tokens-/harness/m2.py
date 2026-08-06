import ref


def check(workdir):
    from metalopt.quant import check_tensor_divisible_by_256, compare_quantization_profiles

    error_raised_count = 0
    dimensions_checked_count = 0
    total = len(ref.TENSOR_TEST_CASES)

    for shape in ref.TENSOR_TEST_CASES:
        should_fail = any(dim % 256 != 0 for dim in shape)
        try:
            check_tensor_divisible_by_256(shape)
            if not should_fail:
                dimensions_checked_count += 1
        except ValueError as e:
            if "not divisible by 256" in str(e):
                if should_fail:
                    error_raised_count += 1
            else:
                pass

    # Evaluate comparison profile
    comp = compare_quantization_profiles(4.5, 6.0, 5.2, 5.1)
    comp_valid = 1 if isinstance(comp, dict) and "size_ratio" in comp else 0

    out = {
        "error_raised": 1.0 if error_raised_count > 0 else 0.0,
        "dimensions_checked": 1.0 if (dimensions_checked_count + error_raised_count) == total else 0.0,
    }
    if comp_valid == 0:
        out["_note"] = "compare_quantization_profiles did not return expected dict structure"
    return out
