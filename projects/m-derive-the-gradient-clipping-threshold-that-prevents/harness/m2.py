import ref

def check(workdir):
    from numerics.overflow import check_precision_overflow
    val_overflow = 70000.0
    res_fp16 = check_precision_overflow(val_overflow, "float16")
    res_bf16 = check_precision_overflow(val_overflow, "bfloat16")

    is_overflow_fp16 = (res_fp16 == float('inf') or res_fp16 != res_fp16)
    is_safe_bf16 = (res_bf16 == val_overflow)

    if is_overflow_fp16 and is_safe_bf16:
        return {"overflow_matched": 1.0}
    return {"overflow_matched": 0.0, "_note": f"fp16 res: {res_fp16}, bf16 res: {res_bf16}"}
