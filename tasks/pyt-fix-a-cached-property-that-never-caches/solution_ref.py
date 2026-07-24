def cached_property(func):
    name = func.__name__

    class CachedProperty:
        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            value = func(obj)
            obj.__dict__[name] = value
            return value

    return CachedProperty()
