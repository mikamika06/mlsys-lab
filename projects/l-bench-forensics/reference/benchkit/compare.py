from .parse import config

IGNORED = {"model_filename", "_source", "_row"}


def differences(a, b):
    """Configuration keys where two rows disagree."""
    ca, cb = config(a), config(b)
    keys = (set(ca) | set(cb)) - IGNORED
    return sorted(k for k in keys if ca.get(k) != cb.get(k))


def controlled(rows, axis):
    """Pairs that differ in `axis` alone: the only comparisons worth quoting."""
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = differences(rows[i], rows[j])
            if d == [axis]:
                out.append((i, j))
    return out


def confounded(rows, axis):
    """Pairs that differ in `axis` and in something else as well."""
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = differences(rows[i], rows[j])
            if axis in d and len(d) > 1:
                out.append((i, j, [x for x in d if x != axis]))
    return out
