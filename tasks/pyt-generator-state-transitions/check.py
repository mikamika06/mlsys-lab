import inspect


def _oracle():
    seen = []
    holder = {}

    def scripted():
        g = holder["g"]
        seen.append(inspect.getgeneratorstate(g))
        yield "first"
        seen.append(inspect.getgeneratorstate(g))
        yield "second"

    g = scripted()
    holder["g"] = g

    seen.append(inspect.getgeneratorstate(g))
    next(g)
    seen.append(inspect.getgeneratorstate(g))
    next(g)
    seen.append(inspect.getgeneratorstate(g))
    try:
        next(g)
    except StopIteration:
        pass
    seen.append(inspect.getgeneratorstate(g))

    return seen


def grade(sol, fx) -> dict:
    try:
        got = sol.generator_states()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if list(got) == _oracle() else 0.0}
