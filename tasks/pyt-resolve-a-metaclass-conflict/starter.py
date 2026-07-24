def resolve_mro_names():
    # TODO: The two base classes use unrelated metaclasses. Creating Result
    # directly triggers TypeError: metaclass conflict.
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

    class Result(Left, Right):
        pass

    return [cls.__name__ for cls in Result.__mro__]
