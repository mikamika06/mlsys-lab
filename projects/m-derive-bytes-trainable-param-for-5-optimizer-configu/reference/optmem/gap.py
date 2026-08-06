def memory_multiplier_gap(full_ft_bytes, lora_bytes):
    if lora_bytes == 0:
        return 0.0
    return float(full_ft_bytes) / float(lora_bytes)
