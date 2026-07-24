class NamedField:
    """A descriptor that discovers its own attribute name automatically.

    `type.__new__` calls `__set_name__(self, owner, name)` on every
    descriptor found in a freshly-created class body, once per binding,
    passing the owning class and the name it was assigned to. That's the
    only reliable way a descriptor can learn its own name — there is no
    other hook for it. Values are stored per-instance under a private
    ("_" + name) key.
    """

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)


def recovered_names(cls):
    """Return {class_attribute_name: name_captured_by___set_name__} for
    every NamedField descriptor defined directly on `cls`."""
    return {
        attr: desc.name
        for attr, desc in vars(cls).items()
        if isinstance(desc, NamedField)
    }
