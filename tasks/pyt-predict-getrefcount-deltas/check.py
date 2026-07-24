import sys


def _oracle():
    def measure(obj):
        return sys.getrefcount(obj) - 1

    x = object()
    start = measure(x)
    values = []

    aliases = [x]
    values.append(measure(x))

    aliases.append(x)
    values.append(measure(x))

    del aliases[0]
    values.append(measure(x))

    del aliases
    values.append(measure(x))

    return [v - start for v in values]


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.predict_refcount_deltas()
        got = list(got)
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
