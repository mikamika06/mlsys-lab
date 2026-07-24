import sys


def dict_vs_slots_size_ratio():
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

    dict_instance = DictBacked()
    slot_instance = Slotted()

    dict_size = sys.getsizeof(dict_instance) + sys.getsizeof(dict_instance.__dict__)
    slot_size = sys.getsizeof(slot_instance)
    return dict_size / slot_size
