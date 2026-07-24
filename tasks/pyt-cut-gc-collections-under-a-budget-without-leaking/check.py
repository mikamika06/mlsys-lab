import gc


def _oracle():
    old_thresholds = gc.get_threshold()
    old_enabled = gc.isenabled()
    events = []

    def callback(phase, info):
        if phase == "stop":
            events.append(info.get("generation", -1))

    gc.callbacks.append(callback)
    try:
        gc.collect()
        before = len(gc.get_objects())

        gc.disable()
        gc.freeze()
        gc.set_threshold(1000000, 1000000, 1000000)

        roots = []
        for _ in range(200):
            a = []
            b = [a]
            a.append(b)
            roots.append(a)
        roots.clear()

        gc.unfreeze()
        gc.collect()
        after = len(gc.get_objects())
        return len(events), before != after
    finally:
        if callback in gc.callbacks:
            gc.callbacks.remove(callback)
        gc.set_threshold(*old_thresholds)
        if old_enabled:
            gc.enable()
        else:
            gc.disable()


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.cut_gc_collections_under_budget()
        got = (int(got[0]), bool(got[1]))
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
