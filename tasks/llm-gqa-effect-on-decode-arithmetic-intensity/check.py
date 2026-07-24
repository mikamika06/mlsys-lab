def _reference(num_heads, head_dim, seq_len, use_gqa):
    kv_heads = num_heads // 2 if use_gqa else num_heads
    ops = num_heads * head_dim * seq_len
    mem_bytes = num_heads * head_dim + 2 * seq_len * kv_heads * head_dim
    return ops / mem_bytes

def grade(sol, fx) -> dict:
    cases = [
        (8, 64, 128, False),
        (8, 64, 128, True),
        (12, 32, 256, False),
        (12, 32, 256, True),
    ]
    max_rel_err = 0.0
    for num_heads, head_dim, seq_len, use_gqa in cases:
        try:
            got = sol.decode_arithmetic_intensity(num_heads, head_dim, seq_len, use_gqa)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(num_heads, head_dim, seq_len, use_gqa)
        rel_err = abs(got - ref) / max(abs(ref), 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": max_rel_err}
