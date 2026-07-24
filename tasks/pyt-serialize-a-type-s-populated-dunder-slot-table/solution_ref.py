DUUNDERS = [
    "__new__", "__init__", "__repr__", "__str__", "__bytes__",
    "__format__", "__lt__", "__le__", "__eq__", "__ne__",
    "__gt__", "__ge__", "__hash__", "__bool__", "__len__",
    "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__call__", "__enter__", "__exit__", "__await__",
    "__aiter__", "__anext__", "__add__", "__sub__", "__mul__",
    "__matmul__", "__truediv__", "__floordiv__", "__mod__",
    "__pow__", "__neg__", "__pos__", "__abs__", "__invert__",
    "__index__"
]


def serialize_dunder_slots(classes):
    result = bytearray()
    for cls in classes:
        mask = 0
        for i, name in enumerate(DUUNDERS):
            if getattr(cls, name, None) is not None:
                mask |= 1 << i
        result.extend(mask.to_bytes(8, "little", signed=False))
    return bytes(result)
