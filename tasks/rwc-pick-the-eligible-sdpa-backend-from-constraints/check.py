def _oracle(head_dim, dtype, mask_type, causal):
    if head_dim % 64 == 0 and dtype in {"float16", "bfloat16"} and mask_type == "causal" and causal:
        return "flash"
    elif head_dim <= 256 and dtype == "float32" and mask_type == "full":
        return "mem_efficient"
    else:
        return "math"

def grade(sol, fx) -> dict:
    cases = [
        (128, "float16", "causal", True),   # flash
        (256, "bfloat16", "causal", True),  # flash
        (64,  "float32", "full", False),    # mem_efficient
        (200, "float32", "full", False),    # mem_efficient
        (512, "float32", "full", False),    # math
        (128, "float16", "full", False),    # math
        (64,  "int8",    "causal", True),   # math
        (256, "float32", "causal", False),  # math
    ]
    ok = 1.0
    for case in cases:
        try:
            got = sol.pick_backend(*case)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(*case)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
