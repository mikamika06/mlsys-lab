def get_format_props(format_name):
    formats = {
        "FP32": {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 23, "max_val": 3.4028235e38, "min_pos": 1.17549435e-38, "ulp_eps": 1.1920929e-07},
        "TF32": {"sign_bits": 1, "exp_bits": 8, "mantissa_bits": 10, "max_val": 3.4028235e38, "min_pos": 1.17549435e-38, "ulp_eps": 9.765625e-04},
        "FP16": {"sign_bits": 1, "exp_bits": 5, "mantissa_bits": 10, "max_val": 65504.0, "min_pos": 6.1035156e-05, "ulp_eps": 0.0009765625}
    }
    if format_name not in formats:
        raise ValueError(f"Unknown format: {format_name}")
    return formats[format_name]
