import math


def _stall(c, total_tokens, budget, alpha, beta, gamma):
    chunks = (total_tokens + c - 1) // c
    prefill = chunks * (alpha * c * c + beta * c)
    decode = chunks * gamma * c
    return decode + max(0.0, prefill - budget)


def _oracle(total_tokens, budget, alpha, beta, gamma, max_chunk):
    best_c = 1
    best_s = float("inf")
    for c in range(1, min(total_tokens, max_chunk) + 1):
        s = _stall(c, total_tokens, budget, alpha, beta, gamma)
        if s < best_s:
            best_s = s
            best_c = c
    return best_c, best_s


def grade(sol, fx) -> dict:
    cases = [
        (128, 5000.0, 0.01, 0.5, 2.0, 64),
        (512, 8000.0, 0.002, 1.0, 3.0, 128),
        (73, 200.0, 0.05, 0.1, 1.5, 40),
        (1024, 12000.0, 0.001, 0.2, 5.0, 256),
        (37, 20.0, 0.1, 0.0, 0.5, 32),
    ]

    ratio = 1.0
    for args in cases:
        try:
            got_c = int(sol.select_chunk_size(*args))
        except Exception:
            return {"size_ratio": float("inf")}

        total_tokens, budget, alpha, beta, gamma, max_chunk = args
        _, optimal_stall = _oracle(*args)
        if got_c < 1 or got_c > min(total_tokens, max_chunk):
            return {"size_ratio": float("inf")}

        got_stall = _stall(got_c, total_tokens, budget, alpha, beta, gamma)
        ratio = max(ratio, got_stall / (optimal_stall + 1e-12))

    return {"size_ratio": ratio}
