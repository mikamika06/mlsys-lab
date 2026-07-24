def _observe(prop_factory):
    values = []

    def get_x(obj):
        return obj._x

    def set_x(obj, value):
        obj._x = value

    def del_x(obj):
        del obj._x

    class Box:
        prop = prop_factory(get_x, set_x, del_x, "doc")

    b = Box()
    b._x = 10

    try:
        values.append(b.prop)
    except Exception as exc:
        values.append(type(exc).__name__)

    try:
        b.prop = 25
        values.append(b.prop)
    except Exception as exc:
        values.append(type(exc).__name__)

    try:
        del b.prop
        values.append(not hasattr(b, "_x"))
    except Exception as exc:
        values.append(type(exc).__name__)

    try:
        values.append(Box.prop is Box.__dict__["prop"])
    except Exception as exc:
        values.append(type(exc).__name__)

    class ReadOnly:
        prop = prop_factory(get_x)

    r = ReadOnly()
    r._x = 3

    try:
        r.prop = 4
        values.append("no error")
    except Exception as exc:
        values.append(type(exc).__name__)

    class WriteOnly:
        prop = prop_factory(None, set_x)

    w = WriteOnly()
    try:
        values.append(w.prop)
    except Exception as exc:
        values.append(type(exc).__name__)

    return values


def grade(sol, fx) -> dict:
    try:
        def get_x(obj):
            return obj._x

        def set_x(obj, value):
            obj._x = value

        def del_x(obj):
            del obj._x

        class Box:
            prop = property(get_x, set_x, del_x, "doc")

        b = Box()
        b._x = 10

        expected = _observe(property)
        got = _observe(sol.property_from_scratch)
        ok = float(got == expected)
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
