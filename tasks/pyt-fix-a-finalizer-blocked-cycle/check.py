import gc
import weakref


def _oracle():
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


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.collect_cycle_with_finalizer()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
