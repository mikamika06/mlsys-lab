class NamedField:
    """A descriptor that discovers its own attribute name automatically via
    __set_name__. Values are stored per-instance under a private
    ("_" + name) key."""

    def __set_name__(self, owner, name):
        raise NotImplementedError('your code here')

    def __get__(self, obj, objtype=None):
        raise NotImplementedError('your code here')

    def __set__(self, obj, value):
        raise NotImplementedError('your code here')


def recovered_names(cls):
    """Return {class_attribute_name: name_captured_by___set_name__} for
    every NamedField descriptor defined directly on `cls`."""
    raise NotImplementedError('your code here')
