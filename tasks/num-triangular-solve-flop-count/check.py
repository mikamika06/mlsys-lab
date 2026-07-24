def _oracle(n):
    count = 0
    for row in range(n):
        for _ in range(row):
            count += 1
    return count


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 4, 8, 17, 64]
    ok = 1.0
    for n in cases:
        try:
            got = sol.triangular_solve_flops(n)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(n):
            ok = 0.0
            break
    return {"exact_match": ok}
