def _oracle(name):
    mapping = {
        "naive": "full",
        "math-SDPA": "full",
        "mem-efficient": "efficient",
        "flash": "efficient"
    }
    return mapping[name]


def grade(sol, fx) -> dict:
    cases = ["naive", "math-SDPA", "mem-efficient", "flash"]
    ok = 1.0
    for name in cases:
        try:
            got = sol.classify_backend(name)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(name):
            ok = 0.0
            break
    return {"exact_match": ok}
