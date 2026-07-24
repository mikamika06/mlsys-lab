class cached_property:
    """
    A NON-DATA descriptor (no ``__set__``) that computes the wrapped
    method's value on first access and caches it in the instance's
    ``__dict__`` under the method's own name, so every later access finds
    it directly there and never calls ``__get__`` (and therefore never
    re-runs the wrapped function) again.
    """

    def __init__(self, func):
        self.func = func
        self.attrname = None

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "cached_property used without calling __set_name__ on it."
            )
        value = self.func(instance)
        instance.__dict__[self.attrname] = value
        return value
