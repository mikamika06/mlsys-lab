import sys


def _oracle_size():
    class Base:
        __slots__ = ("value",)

    class Fixed(Base):
        __slots__ = ()

    obj = Fixed()
    obj.value = 1
    return sys.getsizeof(obj)


def _measure(cls):
    obj = cls()
    obj.value = 1
    if hasattr(obj, "__dict__"):
        return None
    return sys.getsizeof(obj)


def grade(sol, fx) -> dict:
    try:
        class Base:
            __slots__ = ("value",)

        class Broken(Base):
            def double(self):
                return self.value * 2

        class BrokenAgain(Base):
            def triple(self):
                return self.value * 3

        target = _oracle_size()
        fixed1 = sol.restore_slots(Broken)
        fixed2 = sol.restore_slots(BrokenAgain)

        measured = [_measure(fixed1), _measure(fixed2)]
        if any(x is None for x in measured):
            return {"size_ratio": 0.0}

        ratio = min(measured) / target
        return {"size_ratio": float(ratio)}
    except Exception:
        return {"size_ratio": 0.0}
