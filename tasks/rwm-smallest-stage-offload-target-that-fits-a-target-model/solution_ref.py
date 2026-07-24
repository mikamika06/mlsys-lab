def choose_stage_offload(Phi: int, N: int, gpu_memory: int) -> tuple:
    param = 2 * Phi
    grad = 2 * Phi
    opt = 8 * Phi

    candidates = [
        (0, "none"),
        (1, "none"),
        (2, "none"),
        (3, "none"),
        (1, "cpu"),
        (2, "cpu"),
        (3, "cpu"),
    ]

    for stage, offload in candidates:
        gpu_param = param
        gpu_grad = grad
        gpu_opt = opt

        if stage >= 1:
            gpu_opt /= N
        if stage >= 2:
            gpu_grad /= N
        if stage >= 3:
            gpu_param /= N

        if offload == "cpu":
            gpu_opt = 0

        if gpu_param + gpu_grad + gpu_opt <= gpu_memory:
            return stage, offload

    return None
