def compare_recompiles(shapes, dynamic_mode):
    seen = set()
    recompiles = 0
    for shape in shapes:
        if dynamic_mode is True:
            key = tuple(1 if x > 1 else x for x in shape)
        elif dynamic_mode is False:
            key = shape
        else:
            key = tuple(x if x <= 4 else -1 for x in shape)
        if key not in seen:
            seen.add(key)
            recompiles += 1
    return recompiles
