def _oracle(b, s, h, a):
    term1 = 34.0 * s * b * h
    term2 = 5.0 * a * (s ** 2) * b
    return term1 + term2


def _cases():
    return [
        (1, 2048, 4096, 32),
        (8, 512, 768, 12),
        (4, 1024, 2048, 16),
        (16, 128, 1024, 8),
        (2, 4096, 8192, 64),
        (1, 1, 64, 1),
        (32, 256, 512, 8),
    ]


def grade(sol, fx) -> dict:
    worst = 0.0
    for b, s, h, a in _cases():
        ref = _oracle(b, s, h, a)
        try:
            got = float(sol.activation_memory_bytes(b, s, h, a))
        except Exception:
            return {"size_ratio": float("inf")}
        err = abs(got / ref - 1.0)
        worst = max(worst, err)
    return {"size_ratio": worst}
