def mro_names(cls):
    return tuple(c.__name__ for c in cls.__mro__)
