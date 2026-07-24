def _oracle(Phi, N, stage, offload_target):
    param = Phi * 2
    grad = Phi * 2
    optimizer = Phi * 4

    partitioned = 0
    replicated = 0

    if stage == 1:
        partitioned = optimizer
        replicated = param + grad
    elif stage == 2:
        partitioned = optimizer + grad
        replicated = param
    elif stage == 3:
        partitioned = optimizer + grad + param
        replicated = 0

    gpu = replicated + partitioned // N

    offload = 0

    if offload_target in ("optimizer", "optimizer+param"):
        moved = optimizer // N if stage == 1 else optimizer // N
        gpu -= moved
        offload += moved

    if offload_target in ("param", "optimizer+param") and stage == 3:
        moved = param // N
        gpu -= moved
        offload += moved

    return gpu, offload


def grade(sol, fx) -> dict:
    cases = []
    for Phi in [1, 7, 1000, 12345]:
        for N in [1, 2, 4, 8]:
            for stage in [1, 2, 3]:
                for target in ["none", "optimizer", "param", "optimizer+param"]:
                    cases.append((Phi, N, stage, target))

    ok = 1.0
    for case in cases:
        try:
            got = sol.zero_rank_bytes(*case)
        except Exception:
            ok = 0.0
            break
        if tuple(got) != _oracle(*case):
            ok = 0.0
            break
    return {"exact_match": ok}
