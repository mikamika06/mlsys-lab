def compute_ai(flops: int, bytes_transferred: int) -> float:
    if bytes_transferred == 0:
        return float("inf")
    return float(flops) / float(bytes_transferred)


def rank_kernels(kernels: list) -> list:
    scored = []
    for k in kernels:
        ai = compute_ai(k["flops"], k["bytes"])
        scored.append((ai, k["name"]))
    scored.sort(key=lambda x: x[0])
    return [name for _, name in scored]
