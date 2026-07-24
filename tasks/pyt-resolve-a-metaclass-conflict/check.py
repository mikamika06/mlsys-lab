def _oracle():
    class MetaA(type):
        def marker_a(cls):
            return "a"

    class MetaB(type):
        def marker_b(cls):
            return "b"

    class Left(metaclass=MetaA):
        pass

    class Right(metaclass=MetaB):
        pass

    class Combined(MetaA, MetaB):
        pass

    class Result(Left, Right, metaclass=Combined):
        pass

    return [cls.__name__ for cls in Result.__mro__]


def grade(sol, fx) -> dict:
    try:
        got = sol.resolve_mro_names()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == _oracle() else 0.0}
