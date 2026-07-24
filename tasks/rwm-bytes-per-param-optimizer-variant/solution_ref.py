def optimizer_variant(state_dict):
    total_bytes = sum(arr.nbytes for arr in state_dict.values())
    num_params = next(iter(state_dict.values())).size
    bpp = total_bytes / num_params
    if abs(bpp - 8) < 0.5:
        return "adam_fp32"
    elif abs(bpp - 4) < 0.5:
        return "adam_fp16"
    else:
        return "adam_uint8"
