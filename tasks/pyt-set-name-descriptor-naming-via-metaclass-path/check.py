def grade(sol, fx) -> dict:
    try:
        NamedField = sol.NamedField
        FieldMeta = sol.FieldMeta

        class Widget(metaclass=FieldMeta):
            width = NamedField()
            height = NamedField()
            note = "not a field"
            label = NamedField()

        expected_fields = ("width", "height", "label")
        if tuple(Widget._fields) != expected_fields:
            return {"exact_match": 0.0}

        for name in expected_fields:
            desc = Widget.__dict__[name]
            if getattr(desc, "name", None) != name:
                return {"exact_match": 0.0}

        w = Widget()
        w.width = 10
        w.height = 20
        w.label = "box"
        if (w.width, w.height, w.label) != (10, 20, "box"):
            return {"exact_match": 0.0}

        # per-instance storage: a second instance must not share state
        w2 = Widget()
        w2.width = 99
        if w.width == 99 or w2.width != 99:
            return {"exact_match": 0.0}

        class Empty(metaclass=FieldMeta):
            pass

        if tuple(Empty._fields) != ():
            return {"exact_match": 0.0}

        class Other(metaclass=FieldMeta):
            a = NamedField()
            b = NamedField()

        if tuple(Other._fields) != ("a", "b"):
            return {"exact_match": 0.0}

        # building Other/Empty must not disturb Widget's already-built _fields
        if tuple(Widget._fields) != expected_fields:
            return {"exact_match": 0.0}

    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
