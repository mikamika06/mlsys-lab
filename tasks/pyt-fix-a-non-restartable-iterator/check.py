def _ref(n):
    return [i * i for i in range(n)]


def grade(sol, fx) -> dict:
    ok = 1.0
    for n in (0, 1, 4, 6, 10):
        try:
            obj = sol.Squares(n)
            first = list(obj)
            second = list(obj)
        except Exception:
            ok = 0.0
            break
        expected = _ref(n)
        if first != expected or second != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
