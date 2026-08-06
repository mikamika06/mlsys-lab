def compute_bpw(quant_type: str) -> float:
    table = {
        "IQ1_S": 1.3125,
        "IQ2_XXS": 2.25,
        "IQ4_XS": 4.25,
        "TQ1_0": 1.0,
        "TQ2_0": 2.0
    }
    return table[quant_type]
