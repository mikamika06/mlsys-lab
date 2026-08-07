def get_bpw(quant_type):
    bits_map = {
        "F32": 32.0,
        "F16": 16.0,
        "Q8_0": 8.5,
        "Q4_K_M": 4.5
    }
    return bits_map.get(quant_type, 16.0)
