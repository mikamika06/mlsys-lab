import weakref

_GRAVEYARD = []


class Resource:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def __del__(self):
        self.events.append(("finalized", self.name))
        _GRAVEYARD.append(self)          # BUG: resurrects itself


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
