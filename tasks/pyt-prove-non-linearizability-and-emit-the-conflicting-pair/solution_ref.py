import re


def conflicting_pair(spec):
    classes = {}
    try:
        for name, bases in spec:
            if not bases:
                classes[name] = type(name, (), {})
            else:
                classes[name] = type(name, tuple(classes[b] for b in bases), {})
    except TypeError as exc:
        match = re.search(r"bases ([A-Za-z_]\w*), ([A-Za-z_]\w*)", str(exc))
        if match:
            return (match.group(1), match.group(2))
    return None
