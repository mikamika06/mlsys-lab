import gc


def _oracle(n):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        tmp = [[] for _ in range(n)]
        del tmp
        gc.collect()
        return tuple(gc.get_count())
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


def grade(sol, fx) -> dict:
    cases = [0, 1, 10, 1000]
    ok = 1.0
    for n in cases:
        expected = _oracle(n)
        try:
            got = tuple(sol.measure_gc_count(n))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
