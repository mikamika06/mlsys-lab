def derive_kernel_count(p_ops: int, has_reduction: bool, q_ops: int) -> int:
    if not has_reduction:
        return 1 if (p_ops + q_ops) > 0 else 0
    kernels = 1
    if q_ops > 0:
        kernels += 1
    if p_ops > 3:
        kernels += 1
    return min(kernels, p_ops + q_ops + 1)
