def build_recipe():
    return {
        "quant_method": "fp8",
        "weight_dtype": "fp8_e4m3",
        "input_dtype": "fp8_e4m3",
        "targets": ["Linear"]
    }
