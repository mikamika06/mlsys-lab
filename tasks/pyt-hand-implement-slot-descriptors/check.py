def _oracle():
    missing = object()

    class Descriptor:
        def __init__(self, index):
            self.index = index

        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            value = object.__getattribute__(obj, "_storage")[self.index]
            if value is missing:
                raise AttributeError("missing")
            return value

        def __set__(self, obj, value):
            object.__getattribute__(obj, "_storage")[self.index] = value

        def __delete__(self, obj):
            object.__getattribute__(obj, "_storage")[self.index] = missing

    class SlotObject:
        x = Descriptor(0)
        y = Descriptor(1)
        z = Descriptor(2)

        def __init__(self):
            object.__setattr__(self, "_storage", [missing, missing, missing])

        def __getattribute__(self, name):
            if name == "__dict__":
                raise AttributeError("no dictionary")
            return object.__getattribute__(self, name)

    obj = SlotObject()
    obj.x = 10
    obj.y = 20
    obj.z = 30

    values = [obj.x, obj.y, obj.z]

    del obj.y
    try:
        obj.y
        deleted = False
    except AttributeError:
        deleted = True

    try:
        obj.__dict__
        no_dict = False
    except AttributeError:
        no_dict = True

    return (values, no_dict and deleted)


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.hand_slots_roundtrip()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
