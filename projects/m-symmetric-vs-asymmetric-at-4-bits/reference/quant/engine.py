def determine_group_size(case):
    bits = case["target_bits"]
    p = case["total_params"]
    gs = 128 if bits <= 4.0 else 64
    if case["symmetric"]:
        gs = min(gs, p)
    else:
        gs = max(32, gs // 2)
    return gs
