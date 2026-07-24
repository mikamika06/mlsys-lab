def _oracle():
    from itertools import combinations

    class Data:
        def __get__(self, obj, owner):
            return "data"

        def __set__(self, obj, value):
            pass

    class NonData:
        def __get__(self, obj, owner):
            return "nondata"

    def winner(active):
        class Probe:
            pass

        if 1 in active:
            Probe.x = Data()
        elif 3 in active:
            Probe.x = NonData()
        elif 4 in active:
            Probe.x = "class"

        obj = Probe()
        if 2 in active:
            obj.__dict__["x"] = "instance"

        if 6 in active:
            def fallback(self, name):
                return "fallback"
            Probe.__getattr__ = fallback

        value = getattr(obj, "x")
        return {
            "data": 1,
            "instance": 2,
            "nondata": 3,
            "class": 4,
            "fallback": 6,
        }[value]

    providers = [1, 2, 3, 4, 6]
    edges = {x: set() for x in providers}

    for a, b in combinations(providers, 2):
        first = winner({a, b})
        second = b if first == a else a
        edges[first].add(second)

    order = []
    remaining = set(providers)
    while remaining:
        choices = [x for x in remaining if not any(x in edges[y] for y in remaining)]
        choices.sort()
        pick = choices[0]
        order.append(pick)
        remaining.remove(pick)

    class HookProbe:
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    seen = []
    original = HookProbe.__getattribute__

    def traced(self, name):
        seen.append(5)
        return original(self, name)

    HookProbe.__getattribute__ = traced
    try:
        _ = HookProbe()
        obj = HookProbe()
        try:
            obj.missing
        except AttributeError:
            pass
    finally:
        HookProbe.__getattribute__ = original

    return [5] + order


def grade(sol, fx) -> dict:
    try:
        got = sol.emit_attribute_precedence_rule_list()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if list(got) == _oracle() else 0.0}
