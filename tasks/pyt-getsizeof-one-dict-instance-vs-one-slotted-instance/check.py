import sys


def _oracle():
    class DictBacked:
        def __init__(self):
            self.a = 1
            self.b = 2
            self.c = 3

    class Slotted:
        __slots__ = ("a", "b", "c")

        def __init__(self):
            self.a = 1
            self.b = 2
            self.c = 3

    d = DictBacked()
    s = Slotted()
    return (
        sys.getsizeof(d) + sys.getsizeof(d.__dict__)
    ) / sys.getsizeof(s)


def grade(sol, fx) -> dict:
    try:
        got = float(sol.dict_vs_slots_size_ratio())
        ref = float(_oracle())
        if ref == 0:
            score = 0.0
        else:
            score = max(0.0, 1.0 - abs(got - ref) / abs(ref))
    except Exception:
        score = 0.0
    return {"size_ratio": score}
