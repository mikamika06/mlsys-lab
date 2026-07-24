import gc


def collect_cycle_with_finalizer():
    events = []

    class Node:
        def __del__(self):
            events.append("finalized")

    a = Node()
    b = Node()
    a.link = b
    b.link = a

    a = None
    b = None
    gc.collect()

    # TODO: This uses __del__ for cleanup, which does not provide the
    # weakref.finalize lifecycle guarantee required by the task.
    events.append("alive")
    return events
