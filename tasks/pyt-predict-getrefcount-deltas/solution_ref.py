import sys


def predict_refcount_deltas():
    def measure(obj):
        return sys.getrefcount(obj) - 1

    x = object()
    start = measure(x)
    out = []

    aliases = [x]
    out.append(measure(x) - start)

    aliases.append(x)
    out.append(measure(x) - start)

    del aliases[0]
    out.append(measure(x) - start)

    del aliases
    out.append(measure(x) - start)

    return out
