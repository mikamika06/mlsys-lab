import gc


def _oracle():
    old_thresholds = gc.get_threshold()
    old_callbacks = list(gc.callbacks)
    counts = [0, 0, 0]

    def callback(phase, info):
        if phase == "stop":
            gen = info["generation"]
            if 0 <= gen <= 2:
                counts[gen] += int(info["collected"])

    try:
        gc.callbacks[:] = []
        gc.set_threshold(10, 10, 10)
        gc.collect(2)
        gc.callbacks.append(callback)

        holders = []
        for _ in range(6):
            a = []
            b = [a]
            a.append(b)
            holders.append(a)
        holders.clear()

        gc.collect(0)

        for _ in range(20):
            x = []
            y = [x]
            x.append(y)

        gc.collect(1)
        gc.collect(2)

        return tuple(counts)
    finally:
        gc.callbacks[:] = old_callbacks
        gc.set_threshold(*old_thresholds)
        gc.collect(2)


def grade(sol, fx) -> dict:
    try:
        expected = _oracle()
        got = sol.objects_collected_per_generation()
        ok = 1.0 if tuple(got) == tuple(expected) else 0.0
    except Exception:
        ok = 0.0
    return {"exact_match": ok}
