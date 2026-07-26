"""Deterministic C++ ABI model — the MLSYS VIRTUAL ABI (pinned LP64, Itanium).
Pure Python, identical on every machine (unlike ctypes.sizeof, which differs by
OS/compiler). Fixed type sizes + natural alignment + standard struct padding, so
'predict the sizeof / offsets' has ONE right answer everywhere.
"""
# pinned sizes/alignment (LP64: long = 8, pointer = 8) — natural alignment
SIZE = {"bool": 1, "char": 1, "signed char": 1, "unsigned char": 1,
        "short": 2, "unsigned short": 2, "int": 4, "unsigned int": 4,
        "long": 8, "unsigned long": 8, "long long": 8, "unsigned long long": 8,
        "float": 4, "double": 8, "long double": 16, "pointer": 8}
ALIGN = dict(SIZE)


def _sz_al(t):
    if t.endswith("*"):
        return 8, 8  # any pointer
    if t not in SIZE:
        raise KeyError(f"unknown type {t!r}")
    return SIZE[t], ALIGN[t]


def layout(fields, packed=False):
    """fields: list of type names (e.g. ['char','int','double']) in declaration
    order. Returns {size, alignment, offsets} under standard C/C++ layout.
    packed=True disables padding (alignment 1)."""
    offset = 0
    max_align = 1
    offsets = []
    for t in fields:
        sz, al = _sz_al(t)
        if packed:
            al = 1
        pad = (-offset) % al
        offset += pad
        offsets.append(offset)
        offset += sz
        max_align = max(max_align, al)
    tail = (-offset) % max_align            # trailing padding to struct alignment
    total = offset + tail
    return {"size": total, "alignment": max_align, "offsets": offsets}


def sizeof(fields, packed=False):
    return layout(fields, packed)["size"]


def offsetof(fields, i, packed=False):
    return layout(fields, packed)["offsets"][i]
