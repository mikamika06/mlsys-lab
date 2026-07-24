def split_group(eg, names):
    wanted = set(names)
    return eg.split(lambda exc: type(exc).__name__ in wanted)
