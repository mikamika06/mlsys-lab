def capture_class_body_order(names):
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

    body = "\n".join(f"        {name} = {i}" for i, name in enumerate(names))
    exec(f"class Temp(metaclass=Meta):\n{body}\n", {"Meta": Meta}, {})
    return list(Meta.last.assigned)
