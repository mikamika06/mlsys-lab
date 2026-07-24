class NamedField:
    """Descriptor that learns its attribute name via __set_name__ and stores
    each instance's value under a private per-instance attribute."""

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


class FieldMeta(type):
    """Metaclass that collects the declared NamedField attribute names into cls._fields."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._fields = tuple(
            attr_name for attr_name, value in namespace.items()
            if isinstance(value, NamedField)
        )
        return cls
