import weakref


class _RefResource:
    """Independent, known-correct oracle: weakref.finalize, no self-capture."""

    def __init__(self, name, events):
        self.name = name
        self._finalizer = weakref.finalize(self, _RefResource._on_finalize, name, events)

    @staticmethod
    def _on_finalize(name, events):
        events.append(("finalized", name))


def _oracle_lifecycle():
    events = []
    for name in ("A", "B"):
        r = _RefResource(name, events)
        ref = weakref.ref(r)
        holder = [r]
        del r
        holder.clear()
        events.append(("alive_after_drop", name, ref() is not None))
    return events


def grade(sol, fx) -> dict:
    ref = _oracle_lifecycle()
    try:
        got = sol.resurrection_safe_lifecycle()
    except Exception:
        return {"exact_match": 0.0}

    ok = isinstance(got, list) and list(got) == ref
    return {"exact_match": 1.0 if ok else 0.0}
