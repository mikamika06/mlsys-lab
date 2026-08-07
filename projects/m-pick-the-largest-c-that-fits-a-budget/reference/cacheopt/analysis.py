def requires_flash_attention(k_bits, v_bits):
    return k_bits < 16 or v_bits < 16


def asymmetric_penalty_ratio(k_bits, v_bits):
    if k_bits != v_bits:
        return 1.15
    return 1.00
