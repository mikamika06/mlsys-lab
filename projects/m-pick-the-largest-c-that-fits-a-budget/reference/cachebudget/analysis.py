def asymmetric_penalty(k_bits, v_bits):
    base = (k_bits + v_bits) / 2.0
    diff = abs(k_bits - v_bits)
    return float(diff * 0.125 + base * 0.05)

def requires_flash_attention(kv_types):
    return kv_types["k"] != 16 or kv_types["v"] != 16
