import ref


def check(workdir):
    from occupancy.mapping import map_field

    matched = 0
    total = len(ref.FIELDS_TO_SECTIONS)
    for k, v in ref.FIELDS_TO_SECTIONS.items():
        got = map_field(k)
        if got == v:
            matched += 1
    out = {"mapping_matched": float(matched == total)}
    return out
