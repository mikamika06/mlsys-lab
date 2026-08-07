def verify_output_shift(base_val, adapter_out, scale, expected):
    val = base_val + (adapter_out * scale)
    return abs(val - expected) < 1e-5
