"""Reference solution for `pyt-reimplement-getattribute-resolution-order`.

Reimplements the precedence order of `object.__getattribute__`:

  1. data descriptor found in the MRO      (has __get__ AND (__set__ or __delete__))
  2. instance __dict__
  3. non-data descriptor found in the MRO  (has __get__ only)
  4. plain class attribute found in the MRO
  5. type(obj).__getattr__ fallback (searched via the MRO too)
  6. AttributeError

No `getattr`, `super`, or `__getattribute__` shortcuts are used to
resolve `name` -- only direct dict/`__mro__` walks and explicit
descriptor-protocol dunder calls, which is what CPython itself does at
the C level.
"""
from __future__ import annotations


def _is_data_descriptor(value) -> bool:
    t = type(value)
    return hasattr(t, "__get__") and (hasattr(t, "__set__") or hasattr(t, "__delete__"))


def _is_descriptor(value) -> bool:
    return hasattr(type(value), "__get__")


def resolve(obj, name: str):
    objtype = type(obj)

    found_in_class = False
    class_value = None
    for klass in objtype.__mro__:
        if name in klass.__dict__:
            class_value = klass.__dict__[name]
            found_in_class = True
            break

    # 1. data descriptor wins over everything, including the instance dict
    if found_in_class and _is_data_descriptor(class_value):
        return type(class_value).__get__(class_value, obj, objtype)

    # 2. instance __dict__ (only meaningful when no data descriptor claimed it)
    inst_dict = obj.__dict__ if hasattr(obj, "__dict__") else None
    if inst_dict is not None and name in inst_dict:
        return inst_dict[name]

    # 3/4. non-data descriptor, or a plain class attribute
    if found_in_class:
        if _is_descriptor(class_value):
            return type(class_value).__get__(class_value, obj, objtype)
        return class_value

    # 5. __getattr__ fallback, searched through the MRO like everything else
    for klass in objtype.__mro__:
        if "__getattr__" in klass.__dict__:
            return klass.__dict__["__getattr__"](obj, name)

    # 6. nothing found anywhere
    raise AttributeError(name)
