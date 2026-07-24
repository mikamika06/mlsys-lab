import gc


def cut_gc_collections_under_budget():
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
