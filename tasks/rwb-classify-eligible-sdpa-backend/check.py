def _ref(dtype, head_dim, has_attn_mask, is_causal, device_is_cpu):
    if device_is_cpu:
        return "math"
    if has_attn_mask:
        return "math"
    if dtype in ("float16", "bfloat16"):
        if head_dim <= 128:
            return "flash"
        elif head_dim <= 256:
            return "mem_efficient"
    return "math"

def grade(sol, fx) -> dict:
    cases = [
        ("float16", 64, False, True, False),
        ("float32", 128, False, False, False),
        ("bfloat16", 200, False, False, False),
        ("float16", 300, False, False, False),
        ("float16", 100, True, False, False),
        ("float32", 50, False, False, True),   # CPU
    ]
    ok = 1.0
    for args in cases:
        try:
            got = sol.classify_backend(*args)
        except Exception:
            ok = 0.0
            break
        ref = _ref(*args)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
