def _oracle():
    meta_registry = {}

    class Meta(type):
        def __new__(mcls, name, bases, namespace):
            cls = super().__new__(mcls, name, bases, namespace)
            if name != "Root":
                meta_registry[name] = {
                    "name": name,
                    "bases": [b.__name__ for b in bases],
                    "attrs": sorted(
                        key for key in namespace
                        if not key.startswith("_")
                    ),
                }
            return cls

    class Root(metaclass=Meta):
        pass

    class Alpha(Root):
        kind = "a"

    class Beta(Root):
        size = 3

    init_registry = {}

    class InitRoot:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            init_registry[cls.__name__] = {
                "name": cls.__name__,
                "bases": [b.__name__ for b in cls.__bases__],
                "attrs": sorted(
                    key for key in cls.__dict__
                    if not key.startswith("_")
                ),
            }

    class Alpha(InitRoot):
        kind = "a"

    class Beta(InitRoot):
        size = 3

    if meta_registry.keys() != init_registry.keys():
        return {}
    return init_registry


def grade(sol, fx) -> dict:
    try:
        got = sol.build_registry()
        expected = _oracle()
        ok = 1.0 if got == expected else 0.0
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
