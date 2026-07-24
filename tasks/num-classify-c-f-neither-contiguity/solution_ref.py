def classify_contiguity(arr):
    """
    Return "C", "F" or "Neither" based on NumPy's contiguity flags.
    """
    c = arr.flags.c_contiguous
    f = arr.flags.f_contiguous
    if c and not f:
        return "C"
    elif f and not c:
        return "F"
    else:
        return "Neither"
