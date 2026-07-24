def _reference(field_types):
    """
    Compute offsets and total size under natural alignment rules.
    Supported types: char, short, int, long, float, double.
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


def grade(sol, fx) -> dict:
    """
    Grade the learner's compute_struct_layout implementation.
    """
    cases = [
        ["char", "int", "short"],
        ["double", "char", "float", "long"],
        ["short", "short", "int"],
        ["char"] * 5,
        ["int", "char", "double", "short", "long"]
    ]
    ok = 1.0
    for fields in cases:
        try:
            res = sol.compute_struct_layout(list(fields))
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(res, (tuple, list)) or len(res) != 2:
            return {"exact_match": 0.0}
        got_offsets, got_size = res
        ref_offsets, ref_size = _reference(fields)
        if got_offsets != ref_offsets or got_size != ref_size:
            ok = 0.0
            break
    return {"exact_match": ok}
