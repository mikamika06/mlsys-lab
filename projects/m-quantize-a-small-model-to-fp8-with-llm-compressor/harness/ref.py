def get_reference_recipe():
    return {
        "quant_method": "fp8",
        "weight_dtype": "fp8_e4m3",
        "input_dtype": "fp8_e4m3",
        "targets": ["Linear"]
    }


def compute_reference_ratio(orig_size, comp_size):
    return float(comp_size) / float(orig_size)
