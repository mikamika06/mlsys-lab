def resolve_precision(context_stack, op_dtype):
    if not context_stack:
        return op_dtype
    top = context_stack[-1]
    if not top.get("enabled", True):
        return "float32"
    return top.get("dtype", op_dtype)
