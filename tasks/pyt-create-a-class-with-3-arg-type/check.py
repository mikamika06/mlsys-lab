_RefAnimal = type(
    "Animal",
    (object,),
    {"kind": "animal", "speak": lambda self: f"{self.kind} makes a sound"},
)

_RefDog = type(
    "Dog",
    (_RefAnimal,),
    {"kind": "dog", "speak": lambda self: f"{self.kind} barks"},
)


def _own_keys(cls):
    return tuple(
        sorted(k for k in vars(cls) if not (k.startswith("__") and k.endswith("__")))
    )


def _fingerprint(cls):
    mro_names = tuple(c.__name__ for c in cls.__mro__)
    keys = _own_keys(cls)
    inst = cls()
    behavior = (getattr(inst, "kind", None), inst.speak() if hasattr(inst, "speak") else None)
    return (cls.__name__, mro_names, keys, behavior)


def grade(sol, fx) -> dict:
    try:
        out = sol.build_animal_hierarchy()
        got_animal, got_dog = out
    except Exception:
        return {"exact_match": 0.0}

    try:
        if not (isinstance(got_animal, type) and isinstance(got_dog, type)):
            return {"exact_match": 0.0}

        if got_dog.__bases__[0] is not got_animal:
            return {"exact_match": 0.0}

        if _fingerprint(got_animal) != _fingerprint(_RefAnimal):
            return {"exact_match": 0.0}

        if _fingerprint(got_dog) != _fingerprint(_RefDog):
            return {"exact_match": 0.0}
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
