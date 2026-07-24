class _Tagged:
    def __init__(self, source):
        self.source = source


class _DataDesc:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        return _Tagged("data")

    def __set__(self, obj, value):
        pass


class _NonDataDesc:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        return _Tagged("nondata")


def _oracle(accesses, class_dict, instance_dict, descriptor_flags):
    namespace = {}

    for name in class_dict:
        kind = descriptor_flags.get(name, "class")
        if kind == "data":
            namespace[name] = _DataDesc(name)
        elif kind == "nondata":
            namespace[name] = _NonDataDesc(name)
        else:
            namespace[name] = _Tagged("class")

    def __getattr__(self, name):
        return _Tagged("getattr")

    namespace["__getattr__"] = __getattr__

    C = type("OracleClass", (), namespace)
    obj = C()

    for name, value in instance_dict.items():
        obj.__dict__[name] = _Tagged("instance")

    labels = []
    mapping = {
        "data": 0,
        "instance": 1,
        "nondata": 2,
        "class": 3,
        "getattr": 4,
    }

    for name in accesses:
        value = getattr(obj, name)
        labels.append(mapping[value.source])

    return labels


def grade(sol, fx) -> dict:
    cases = [
        (
            ["a", "b", "c", "d", "e"],
            ["a", "b", "c", "d"],
            {"a": 1, "b": 2},
            {"a": "data", "b": "nondata", "c": "class", "d": "data"},
        ),
        (
            ["shadow", "plain", "missing", "nd"],
            ["shadow", "plain", "nd"],
            {"shadow": 99, "nd": 100},
            {"shadow": "data", "plain": "class", "nd": "nondata"},
        ),
        (
            ["x"] * 10 + ["y", "z", "q"],
            ["x", "y"],
            {"x": 5, "z": 7},
            {"x": "nondata", "y": "data"},
        ),
    ]

    ok = 1.0
    for args in cases:
        expected = _oracle(*args)
        try:
            got = sol.predict_storage_wins(*args)
        except Exception:
            ok = 0.0
            break
        if list(got) != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
