import gc


def objects_collected_per_generation():
    old_thresholds = gc.get_threshold()
    old_callbacks = list(gc.callbacks)
    counts = [0, 0, 0]

    def callback(phase, info):
        if phase == "stop":
            generation = info["generation"]
            if 0 <= generation <= 2:
                counts[generation] += int(info["collected"])

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
