import gc
import weakref


def _oracle(keep):
    class Value:
        pass

    cache = weakref.WeakValueDictionary()
    refs = {}
    for key in range(10):
        value = Value()
        cache[key] = value
        refs[key] = value

    for key in list(refs):
        if key not in keep:
            del refs[key]

    gc.collect()
    return sorted(cache.keys())


def grade(sol, fx) -> dict:
    cases = [
        [],
        [0],
        [1, 4, 7],
        [2, 3, 5, 8, 9],
        list(range(10)),
    ]

    ok = 1.0
    for keep in cases:
        try:
            got = sol.cache_surviving_keys(list(keep))
        except Exception:
            ok = 0.0
            break

        if got != _oracle(list(keep)):
            ok = 0.0
            break

    return {"exact_match": ok}
