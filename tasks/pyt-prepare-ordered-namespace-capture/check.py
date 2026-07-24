def _oracle(names):
    class RecordingDict(dict):
        def __init__(self):
            super().__init__()
            self.assigned = []

        def __setitem__(self, key, value):
            if key not in {"__module__", "__qualname__"}:
                self.assigned.append(key)
            super().__setitem__(key, value)

    class Meta(type):
        last = None

        @classmethod
        def __prepare__(mcls, name, bases):
            mapping = RecordingDict()
            mcls.last = mapping
            return mapping

    body = "\n".join(f"    {name} = {i}" for i, name in enumerate(names))
    source = f"class Temp(metaclass=Meta):\n{body}\n"
    exec(source, {"Meta": Meta}, {})
    return list(Meta.last.assigned)


def grade(sol, fx) -> dict:
    cases = [
        ["a", "b", "c"],
        ["first", "second", "first"],
        ["z", "x", "z", "middle", "z"],
        ["item0", "item1", "item0", "item2", "item1"],
    ]
    ok = 1.0
    for names in cases:
        try:
            got = sol.capture_class_body_order(list(names))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(names):
            ok = 0.0
            break
    return {"exact_match": ok}
