def compute_struct_layout(field_types):
    """
    Compute field offsets and total struct size under natural alignment rules.
    
    Supported types: 'char', 'short', 'int', 'long', 'float', 'double'.
    """
    sizes = {
        "char": 1,
        "short": 2,
        "int": 4,
        "long": 8,
        "float": 4,
        "double": 8
    }
    offsets = []
    cur = 0
    max_align = 1
    for t in field_types:
        sz = sizes[t]
        align = sz
        if align > max_align:
            max_align = align
        # Align current offset
        if cur % align != 0:
            cur += align - (cur % align)
        offsets.append(cur)
        cur += sz
    # Pad struct size to multiple of max_align
    if cur % max_align != 0:
        cur += max_align - (cur % max_align)
    return offsets, cur
