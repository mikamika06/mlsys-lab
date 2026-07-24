class NamedField:
    """Descriptor that learns its attribute name via __set_name__ and stores
    each instance's value under a private per-instance attribute."""

    def __set_name__(self, owner, name):
        raise NotImplementedError('your code here')

    def __get__(self, obj, objtype=None):
        raise NotImplementedError('your code here')

    def __set__(self, obj, value):
        raise NotImplementedError('your code here')


class FieldMeta(type):
    """Metaclass that collects the declared NamedField attribute names into cls._fields."""

    def __new__(mcs, name, bases, namespace):
        raise NotImplementedError('your code here')
