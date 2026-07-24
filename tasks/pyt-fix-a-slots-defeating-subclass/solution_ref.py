def restore_slots(cls):
    namespace = dict(vars(cls))
    namespace.pop("__dict__", None)
    namespace.pop("__weakref__", None)
    namespace.pop("__classcell__", None)
    namespace["__slots__"] = ()
    return type(cls.__name__, cls.__bases__, namespace)
