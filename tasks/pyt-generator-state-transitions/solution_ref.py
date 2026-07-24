import inspect


def generator_states():
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
