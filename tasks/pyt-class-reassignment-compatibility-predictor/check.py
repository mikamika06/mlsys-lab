def _oracle(pairs):
    result = []
    for source, target in pairs:
        obj = source()
        try:
            obj.__class__ = target
            result.append(True)
        except TypeError:
            result.append(False)
    return result


def grade(sol, fx) -> dict:
    class PlainA:
        pass

    class PlainB:
        pass

    class PlainChild(PlainA):
        pass

    class SlotsEmpty:
        __slots__ = ()

    class SlotsEmptyOther:
        __slots__ = ()

    class SlotsValue:
        __slots__ = ("value",)

    class SlotsValueOther:
        __slots__ = ("value",)

    class SlotsDifferent:
        __slots__ = ("other",)

    class DictSlots:
        __slots__ = ("__dict__", "value")

    pairs = [
        (PlainA, PlainB),
        (PlainA, PlainChild),
        (PlainChild, PlainA),
        (SlotsEmpty, SlotsEmptyOther),
        (SlotsValue, SlotsValueOther),
        (SlotsEmpty, SlotsValue),
        (SlotsValue, PlainA),
        (SlotsValue, SlotsDifferent),
        (DictSlots, PlainA),
    ]

    expected = _oracle(pairs)

    try:
        got = sol.predict_class_reassignment(pairs)
        got = list(got)
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
