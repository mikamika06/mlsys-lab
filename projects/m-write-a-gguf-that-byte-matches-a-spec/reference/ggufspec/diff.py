"""Diff implementation."""

def structural_diff(b1, b2):
    changed = b1 != b2
    details = []
    if len(b1) != len(b2):
        details.append(("size_diff", len(b1), len(b2)))
    return {"changed": changed, "details": details}
