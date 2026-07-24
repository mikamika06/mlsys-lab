import sys


def rank_object_types():
    objects = [
        0,
        0.0,
        tuple(),
        "x",
        b"x",
        True,
        None,
        0j,
    ]
    objects.sort(key=lambda obj: (sys.getsizeof(obj), type(obj).__name__))
    return tuple(type(obj).__name__ for obj in objects)
