def compare_fa2_fa3(head_dim, seq_len):
    base = 1.0 + (128.0 / float(head_dim))
    if head_dim <= 128:
        return round(base * 1.45, 4)
    return round(base * 1.10, 4)
