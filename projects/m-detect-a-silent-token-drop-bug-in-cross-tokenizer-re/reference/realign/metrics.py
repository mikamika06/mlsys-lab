def compute_byte_exact_fraction(original_bytes, realigned_bytes):
    if not original_bytes and not realigned_bytes:
        return 1.0
    matches = sum(1 for a, b in zip(original_bytes, realigned_bytes) if a == b)
    max_len = max(len(original_bytes), len(realigned_bytes))
    if max_len == 0:
        return 1.0
    return float(matches) / float(max_len)
