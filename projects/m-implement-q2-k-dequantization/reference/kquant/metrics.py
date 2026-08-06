def calculate_kquant_bpw(quant_type: str) -> float:
    """Calculates exact theoretical bits per weight for a K-quant superblock type."""
    sizes = {
        "Q2_K": 2 + 128 / 256,
        "Q3_K": 3 + 112 / 256,
        "Q4_K": 4 + 144 / 256,
        "Q5_K": 5 + 176 / 256,
        "Q6_K": 6 + 210 / 256,
    }
    q = quant_type.upper()
    if q not in sizes:
        raise ValueError(f"Unknown quant_type: {quant_type}")
    return sizes[q]
