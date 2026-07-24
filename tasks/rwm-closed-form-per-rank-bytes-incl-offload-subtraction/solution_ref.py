def zero_rank_bytes(Phi, N, stage, offload_target):
    param = Phi * 2
    grad = Phi * 2
    optimizer = Phi * 4

    if stage == 1:
        gpu = param + grad + optimizer // N
    elif stage == 2:
        gpu = param + (grad + optimizer) // N
    else:
        gpu = (param + grad + optimizer) // N

    offload = 0

    if offload_target in ("optimizer", "optimizer+param"):
        moved = optimizer // N
        gpu -= moved
        offload += moved

    if offload_target in ("param", "optimizer+param") and stage == 3:
        moved = param // N
        gpu -= moved
        offload += moved

    return gpu, offload
