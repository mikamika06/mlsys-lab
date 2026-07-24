def resolve_mro_names():
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
