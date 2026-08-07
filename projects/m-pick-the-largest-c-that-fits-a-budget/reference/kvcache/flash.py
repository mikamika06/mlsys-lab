def requires_flash_attention(quant_type):
    qt = quant_type.lower()
    return "q4" in qt or "q8" in qt or "iq" in qt
