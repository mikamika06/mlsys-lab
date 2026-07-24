def _oracle(op_names):
    fusible = {"add", "mul", "relu", "sigmoid", "broadcast"}
    return [op in fusible for op in op_names]


def grade(sol, fx) -> dict:
    cases = [
        (["add", "mm", "relu"],),
        (["conv", "bmm", "nonzero"],),
        (["add", "mul", "sigmoid", "broadcast"],),
        (["reduction", "mm", "add"],),
        ([],),
    ]
    ok = 1.0
    for case in cases:
        try:
            got = sol.tag_ops(list(case[0]))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(case[0])
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
