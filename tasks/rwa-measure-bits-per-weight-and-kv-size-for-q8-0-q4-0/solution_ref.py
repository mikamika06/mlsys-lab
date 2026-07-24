def measure_qkv() -> tuple[float, float, float, float]:
    """
    Returns:
        bpw_q8   – bits per weight for Q8_0
        bpw_q4   – bits per weight for Q4_0
        kv_ratio_q8 – KV size ratio for Q8_0 relative to FP16
        kv_ratio_q4 – KV size ratio for Q4_0 relative to FP16
    """
    block_size = 32
    bpw_q8 = (34 * 8) / block_size
    bpw_q4 = (18 * 8) / block_size
    kv_ratio_q8 = 34 / (block_size * 2)
    kv_ratio_q4 = 18 / (block_size * 2)
    return bpw_q8, bpw_q4, kv_ratio_q8, kv_ratio_q4
