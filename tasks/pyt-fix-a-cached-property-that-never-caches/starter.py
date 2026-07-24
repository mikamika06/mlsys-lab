def cached_property(func):
    name = func.__name__

    class CachedProperty:
        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            value = func(obj)
            obj.__dict__[name] = value
            return value

        # TODO: This makes the descriptor a data descriptor.
        # Because __set__ exists, the instance dictionary entry above
        # is ignored during future lookups and the function recomputes.
        def __set__(self, obj, value):
            obj.__dict__[name] = value

    return CachedProperty()
