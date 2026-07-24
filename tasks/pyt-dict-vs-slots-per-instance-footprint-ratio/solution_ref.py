import sys


def instance_footprint_ratio(attrs: dict) -> float:
    names = tuple(attrs.keys())

    DictClass = type("DictClass", (), {})
    SlotsClass = type(
        "SlotsClass",
        (),
        {"__slots__": names},
    )

    dict_obj = DictClass()
    slots_obj = SlotsClass()

    for key, value in attrs.items():
        setattr(dict_obj, key, value)
        setattr(slots_obj, key, value)

    dict_size = sys.getsizeof(dict_obj) + sys.getsizeof(dict_obj.__dict__)
    slots_size = sys.getsizeof(slots_obj)

    return float(dict_size / slots_size)
