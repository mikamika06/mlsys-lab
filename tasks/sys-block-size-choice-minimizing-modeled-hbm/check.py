import math


def _traffic(M, N, D, Br, Bc):
    if Br < 1 or Bc < 1:
        return None
    if Br > N or Bc > N:
        return None
    if Br * D + Bc * D + Br * Bc > M:
        return None
    return 2 * N * D + 2 * math.ceil(N / Br) * math.ceil(N / Bc) * Bc * D


def _oracle(M, N, D):
    best = None
    for br in range(1, N + 1):
        for bc in range(1, N + 1):
            t = _traffic(M, N, D, br, bc)
            if t is not None and (best is None or t < best):
                best = t
    return best


def grade(sol, fx) -> dict:
    cases = [
        (128, 64, 8),
        (256, 128, 16),
        (512, 96, 12),
        (1024, 256, 32),
        (2048, 192, 64),
    ]
    worst = 1.0
    for M, N, D in cases:
        try:
            br, bc = sol.choose_block_size(M, N, D)
            got = _traffic(M, N, D, int(br), int(bc))
            ref = _oracle(M, N, D)
        except Exception:
            return {"modeled_mem_access": float("inf")}
        if got is None or ref is None:
            return {"modeled_mem_access": float("inf")}
        worst = max(worst, got / ref)
    return {"modeled_mem_access": worst}
