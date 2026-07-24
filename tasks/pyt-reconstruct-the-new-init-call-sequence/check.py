def _oracle_sequence():
    events = []

    class Meta(type):
        def __new__(mcls, name, bases, ns):
            events.append((name, "__new__"))
            return super().__new__(mcls, name, bases, ns)

    class Root(metaclass=Meta):
        def __new__(cls, *args, **kwargs):
            events.append((cls.__name__, "__new__"))
            return super().__new__(cls)

        def __init__(self):
            events.append((self.__class__.__name__, "__init__"))

    class Branch(Root):
        def __new__(cls, *args, **kwargs):
            events.append((cls.__name__, "__new__"))
            return super().__new__(cls)

    class Leaf(Branch):
        def __init__(self):
            events.append((self.__class__.__name__, "__init__"))
            super().__init__()

    events.clear()
    Leaf()
    return events


def grade(sol, fx) -> dict:
    try:
        got = sol.construction_sequence()
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == _oracle_sequence() else 0.0}
