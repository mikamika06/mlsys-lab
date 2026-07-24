import numpy as np


def _oracle(Phi, N, gpu_memory):
    p = np.int64(Phi)
    n = np.int64(N)

    param = 2 * p
    grad = 2 * p
    opt = 8 * p

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
            gpu_opt = gpu_opt / n
        if stage >= 2:
            gpu_grad = gpu_grad / n
        if stage >= 3:
            gpu_param = gpu_param / n

        if offload == "cpu":
            gpu_opt = 0

        per_gpu = gpu_param + gpu_grad + gpu_opt
        if per_gpu <= gpu_memory:
            return (stage, offload)

    return None


def grade(sol, fx) -> dict:
    cases = [
        (1_000_000, 1, 20_000_000),
        (1_000_000_000, 8, 3_000_000_000),
        (7_000_000_000, 16, 1_500_000_000),
        (500_000_000, 4, 900_000_000),
        (12_000_000_000, 32, 1_000_000_000),
        (33_000_000, 2, 100_000_000),
    ]

    rng = np.random.default_rng(7)
    for _ in range(20):
        phi = int(rng.integers(1_000_000, 20_000_000_000))
        n = int(rng.choice([1, 2, 4, 8, 16, 32]))
        memory = int(rng.integers(100_000_000, 20_000_000_000))
        cases.append((phi, n, memory))

    ok = 1.0
    for case in cases:
        try:
            got = sol.choose_stage_offload(*case)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(*case):
            ok = 0.0
            break

    return {"exact_match": ok}
