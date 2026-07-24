import weakref


class Resource:
    def __init__(self, name, events):
        self.name = name
        # The callback must close over plain data only (name, events) --
        # never `self` -- so nothing keeps the object alive once its real
        # refcount hits zero. weakref.finalize guarantees this fires exactly
        # once, no matter what.
        self._finalizer = weakref.finalize(self, Resource._on_finalize, name, events)

    @staticmethod
    def _on_finalize(name, events):
        events.append(("finalized", name))


def resurrection_safe_lifecycle() -> list:
    events = []
    for name in ("A", "B"):
        r = Resource(name, events)
        ref = weakref.ref(r)
        holder = [r]
        del r
        holder.clear()
        events.append(("alive_after_drop", name, ref() is not None))
    return events
