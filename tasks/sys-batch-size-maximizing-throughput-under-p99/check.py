def _oracle(max_batch, slo_ms, fixed_ms, per_item_ms, jitter_ms):
    best_b = 1
    best_t = -1.0
    found = False
    for b in range(1, max_batch + 1):
        latency = fixed_ms + per_item_ms * b + jitter_ms * (b ** 2)
        if latency <= slo_ms:
            found = True
            throughput = 1000.0 * b / latency
            if throughput > best_t:
                best_t = throughput
                best_b = b
    if not found:
        return 1
    return best_b


def _throughput(b, fixed_ms, per_item_ms, jitter_ms):
    latency = fixed_ms + per_item_ms * b + jitter_ms * (b ** 2)
    return 1000.0 * b / latency


def grade(sol, fx) -> dict:
    cases = [
        (64, 50.0, 2.0, 0.5, 0.005),
        (256, 100.0, 5.0, 0.2, 0.001),
        (32, 20.0, 1.0, 0.7, 0.02),
        (128, 30.0, 3.0, 0.1, 0.01),
        (8, 0.5, 2.0, 1.0, 0.1),
    ]
    worst = 1.0
    for args in cases:
        try:
            got = int(sol.max_throughput_batch_size(*args))
        except Exception:
            return {"size_ratio": float("inf")}
        oracle = _oracle(*args)
        opt_t = _throughput(oracle, args[2], args[3], args[4])
        got_t = _throughput(got, args[2], args[3], args[4])
        if got < 1 or got > args[0]:
            return {"size_ratio": float("inf")}
        if got_t <= 0:
            return {"size_ratio": float("inf")}
        worst = max(worst, opt_t / got_t)
    return {"size_ratio": worst}
