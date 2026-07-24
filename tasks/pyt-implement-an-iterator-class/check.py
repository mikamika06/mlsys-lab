def _oracle_sequence(start):
    return list(iter(range(start, 0, -1)))


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 5, 10, 25]
    ok = 1.0
    for start in cases:
        try:
            obj = sol.CountdownIterator(start)
            if iter(obj) is not obj:
                ok = 0.0
                break
            got = []
            while True:
                try:
                    got.append(next(obj))
                except StopIteration:
                    break
            if got != _oracle_sequence(start):
                ok = 0.0
                break
            try:
                next(obj)
                ok = 0.0
                break
            except StopIteration:
                pass
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
