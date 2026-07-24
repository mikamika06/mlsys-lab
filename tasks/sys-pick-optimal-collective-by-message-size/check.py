import math


def _oracle(message_sizes, processes, alpha, beta):
    result = []
    for m in message_sizes:
        tree = (
            alpha * math.ceil(math.log2(processes))
            + beta * m * (processes - 1) / processes
        )
        ring = (
            alpha * (processes - 1)
            + beta * m * (2 * (processes - 1)) / processes
        )
        result.append("tree" if tree <= ring else "ring")
    return result


def grade(sol, fx) -> dict:
    cases = [
        ([1, 64, 4096, 1048576], 8, 10.0, 0.001),
        ([32, 128, 512, 2048], 16, 2.0, 0.01),
        ([0, 100000, 5000000], 4, 5.0, 0.0001),
        ([10, 1000, 100000], 32, 1.0, 0.0005),
        (range(20), 2, 3.5, 0.02),
    ]

    ok = 1.0
    for sizes, processes, alpha, beta in cases:
        try:
            got = sol.pick_collective(list(sizes), processes, alpha, beta)
        except Exception:
            ok = 0.0
            break
        expected = _oracle(list(sizes), processes, alpha, beta)
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
