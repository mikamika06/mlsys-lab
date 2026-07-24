import gc
import weakref


def collect_cycle_with_finalizer():
    events = []

    class Node:
        pass

    a = Node()
    b = Node()
    a.link = b
    b.link = a

    def cleanup():
        events.append("finalized")

    finalizer = weakref.finalize(a, cleanup)

    a = None
    b = None
    gc.collect()

    events.append("dead" if not finalizer.alive else "alive")
    return events
