def is_small_int(n: int) -> bool:
    """
    Return True iff the integer n lies in CPython's small‑int cache [-5,256].
    This matches the behaviour of `a is b` for two separately created ints.
    """
    return -5 <= n <= 256
