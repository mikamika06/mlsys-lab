def validate_mask_dispatch(is_causal, mask_tensor_present):
    if is_causal and mask_tensor_present:
        return "conflict"
    if is_causal:
        return "flash_causal"
    if mask_tensor_present:
        return "math_masked"
    return "flash_standard"
