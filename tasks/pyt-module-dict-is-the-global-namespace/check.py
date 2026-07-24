def _oracle_probe():
    namespace = _oracle_probe.__globals__
    namespace["ARENA_GLOBAL"] = "initial"
    same_object = namespace is _oracle_probe.__globals__
    before = namespace["ARENA_GLOBAL"]
    namespace["ARENA_GLOBAL"] = "mutated"
    after = ARENA_GLOBAL
    return same_object, before, after


def grade(sol, fx) -> dict:
    try:
        expected = _oracle_probe()

        sol_namespace = sol.module_globals_probe.__globals__
        sol_namespace["ARENA_GLOBAL"] = "initial"
        got = sol.module_globals_probe()
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
