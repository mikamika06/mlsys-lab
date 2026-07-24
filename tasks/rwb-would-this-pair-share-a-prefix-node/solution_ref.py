def shares_prefix(a, b):
    """
    Return True iff both sequences are non‑empty and share the same first token.
    """
    if not a or not b:
        return False
    return a[0] == b[0]
