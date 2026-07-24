import gc


def classify_gc_tracking(objects):
    return [gc.is_tracked(obj) for obj in objects]
