class Clamped:
    """Descriptor that is SUPPOSED to keep a public attribute clamped to
    [lo, hi], stored per-instance under a private name (self.private_name).

    BUG: only __get__ is implemented. Without a __set__, this is a
    NON-DATA descriptor, so `obj.level = x` doesn't call this class at
    all — it just writes `x` straight into `obj.__dict__['level']`. From
    then on, plain attribute lookup finds that instance-dict entry BEFORE
    it ever considers this descriptor, so the clamp is silently bypassed
    and never enforced again for that instance.
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


class Widget:
    level = Clamped("level")

    def __init__(self, level=0):
        self.level = level
