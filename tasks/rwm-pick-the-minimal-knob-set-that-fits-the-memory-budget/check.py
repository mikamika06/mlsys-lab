def _peak_memory(param_bytes, activation_bytes, knobs):
    p = float(param_bytes)
    a = float(activation_bytes)
    if "param-offload" in knobs:
        p *= 0.2
    if "checkpoint" in knobs:
        a *= 0.4
    if "activation-offload" in knobs:
        a *= 0.1
    return p + a


def _oracle(param_bytes, activation_bytes, budget_bytes):
    candidates = [
        (),
        ("param-offload",),
        ("checkpoint",),
        ("activation-offload",),
        ("param-offload", "checkpoint"),
        ("param-offload", "activation-offload"),
        ("checkpoint", "activation-offload"),
        ("param-offload", "checkpoint", "activation-offload"),
    ]
    for candidate in candidates:
        if _peak_memory(param_bytes, activation_bytes, candidate) <= budget_bytes:
            return candidate
    return candidates[-1]


def grade(sol, fx) -> dict:
    cases = []
    for p in [0, 100, 512, 4096, 1000000]:
        for a in [0, 50, 300, 8192, 500000]:
            for b in [0, 100, 500, 10000, 2000000]:
                cases.append((p, a, b))

    state = 12345
    for _ in range(100):
        state = (1103515245 * state + 12345) & 0x7fffffff
        p = state % 2000000
        state = (1103515245 * state + 12345) & 0x7fffffff
        a = state % 2000000
        state = (1103515245 * state + 12345) & 0x7fffffff
        b = state % 2500000
        cases.append((p, a, b))

    ok = 1.0
    for p, a, b in cases:
        expected = _oracle(p, a, b)
        try:
            got = tuple(sol.pick_knobs(p, a, b))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
