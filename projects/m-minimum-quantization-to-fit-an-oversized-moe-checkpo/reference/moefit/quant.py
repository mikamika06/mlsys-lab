from moefit.memory import calculate_memory_bytes

def find_min_quant_bits(spec, limit_bytes):
    allowed_bits = [2, 3, 4, 5, 6, 8, 16]
    for b in allowed_bits:
        if calculate_memory_bytes(spec, b) <= limit_bytes:
            return b
    return 16
