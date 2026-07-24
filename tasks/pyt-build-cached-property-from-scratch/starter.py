class cached_property:
    """
    A NON-DATA descriptor (no ``__set__``) that computes the wrapped
    method's value on first access and caches it in the instance's
    ``__dict__`` under the method's own name, so every later access finds
    it directly there and never calls ``__get__`` (and therefore never
    re-runs the wrapped function) again.
    """

    def __init__(self, func):
        raise NotImplementedError('your code here')

    def __set_name__(self, owner, name):
        raise NotImplementedError('your code here')

    def __get__(self, instance, owner=None):
        raise NotImplementedError('your code here')
