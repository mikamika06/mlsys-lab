class Clamped:
    """Descriptor that keeps a public attribute clamped to [lo, hi], stored
    per-instance under a private name (self.private_name).

    Implementing BOTH __get__ and __set__ makes this a DATA descriptor, so
    Python's attribute lookup always calls it — it takes priority over
    anything sitting in the instance's own __dict__ under the same public
    name, even if that entry was written directly (e.g. `obj.__dict__['x']
    = ...`) rather than through normal attribute assignment.
    """

    def __init__(self, name, lo=0, hi=100):
        self.public_name = name
        self.private_name = "_" + name
        self.lo = lo
        self.hi = hi

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, self.lo)

    def __set__(self, obj, value):
        obj.__dict__[self.private_name] = max(self.lo, min(self.hi, value))


class Widget:
    level = Clamped("level")

    def __init__(self, level=0):
        self.level = level
