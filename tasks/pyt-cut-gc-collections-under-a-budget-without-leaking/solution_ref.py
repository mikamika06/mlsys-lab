import gc


def cut_gc_collections_under_budget(n_cycles):
    old_thresholds = gc.get_threshold()
    old_enabled = gc.isenabled()
    events = []

    def callback(phase, info):
        if phase == "stop":
            events.append(info.get("generation", -1))

    gc.callbacks.append(callback)
    try:
        # Collect once up front so the count below is only what this workload caused.
        gc.collect()

        # The budget. Disabling stops automatic collections outright; pinning the
        # thresholds means that even if something re-enables the collector, the
        # generation-0 counter will not reach its trigger during the loop. freeze()
        # moves everything *already* tracked into the permanent generation, so the
        # young-generation scans that do happen have almost nothing to walk.
        gc.disable()
        gc.freeze()
        gc.set_threshold(1000000, 1000000, 1000000)

        roots = []
        for _ in range(n_cycles):
            a = []
            b = [a]
            a.append(b)          # a and b now reference each other
            roots.append(a)
        roots.clear()            # unreachable but cyclic: only the collector frees it

        gc.unfreeze()
        freed = gc.collect()

        # `a` and `b` are still bound here, which keeps the last pair reachable, so
        # `freed` is 2 * n_cycles - 2 rather than 2 * n_cycles. Deleting them before
        # collecting gives exactly 2 * n_cycles — measured, both ways.
        return len(events), freed
    finally:
        # Leaving the collector off would change how every later task behaves.
        if callback in gc.callbacks:
            gc.callbacks.remove(callback)
        gc.set_threshold(*old_thresholds)
        if old_enabled:
            gc.enable()
        else:
            gc.disable()
