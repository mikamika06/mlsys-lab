def _ref(weight_shape, scale_shape):
    w = tuple(weight_shape)
    s = tuple(scale_shape)
    if len(s) == 0 or (len(s) == 1 and s[0] == 1):
        return ("per_tensor", None)
    if len(s) == 1 and s[0] == w[0]:
        return ("per_channel", None)
    if len(s) == 1:
        group_size = w[0] // s[0]
        if w[0] % s[0] == 0 and group_size > 1:
            return ("per_group", group_size)
    raise ValueError("Unsupported shape")

def grade(sol, fx):
    cases = [
        ((64, 3, 7, 7), (1,)),
        ((128, 64, 3, 3), (128,)),
        ((256, 128, 3, 3), (32,)),
        ((10,), (1,)),
        ((20, 5), (20,)),
        ((30, 5), (6,))
    ]
    ok = 1.0
    for wshape, sshape in cases:
        try:
            got = sol.classify_quant_granularity(wshape, sshape)
            ref = _ref(wshape, sshape)
            if got != ref:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
