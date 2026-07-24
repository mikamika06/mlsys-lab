def _oracle(fn):
    names = fn.__code__.co_freevars
    cells = fn.__closure__
    if not cells:
        return []
    return [(name, cell.cell_contents) for name, cell in zip(names, cells)]


def _make_cases():
    def outer_number():
        value = 42
        def inner():
            return value
        return inner

    def outer_multiple():
        left = "alpha"
        right = [1, 2, 3]
        def inner():
            return left, right
        return inner

    def outer_no_closure():
        def inner(x):
            return x + 1
        return inner

    return [
        outer_number(),
        outer_multiple(),
        outer_no_closure(),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for fn in _make_cases():
        try:
            got = sol.reconstruct_closure(fn)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(fn):
            ok = 0.0
            break
    return {"exact_match": ok}
