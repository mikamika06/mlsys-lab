def struct_diff(a, b):
    diffs = []
    if a.get("version") != b.get("version"):
        diffs.append(("version", a.get("version"), b.get("version")))
    return diffs
