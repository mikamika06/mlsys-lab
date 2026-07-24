def pick_knobs(param_bytes, activation_bytes, budget_bytes):
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
        p = float(param_bytes)
        a = float(activation_bytes)
        if "param-offload" in candidate:
            p *= 0.2
        if "checkpoint" in candidate:
            a *= 0.4
        if "activation-offload" in candidate:
            a *= 0.1
        if p + a <= budget_bytes:
            return candidate

    return candidates[-1]
