import gc


def measure_gc_count(n):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        tmp = [[] for _ in range(n)]
        del tmp
        gc.collect()
        return tuple(gc.get_count())
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
