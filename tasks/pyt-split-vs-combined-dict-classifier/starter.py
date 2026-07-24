def is_split_dict(d: dict) -> bool:
    """
    Return True if `d` is currently represented internally by CPython as
    a split-table dict (values stored separately from a shared key
    table), False if it is a normal combined dict.
    """
    raise NotImplementedError('your code here')
