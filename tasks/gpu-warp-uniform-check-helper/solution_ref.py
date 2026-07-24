def all_lanes_agree(pred):
    """
    Return True iff every element in pred has the same truth‑value.
    Handles any iterable of booleans or integers; returns False for an empty input.
    """
    try:
        it = iter(pred)
    except TypeError:
        raise TypeError("Input must be an iterable")
    first = None
    first_seen = False
    for x in it:
        if not first_seen:
            first = x
            first_seen = True
        elif x != first:
            return False
    return first_seen  # empty -> False, otherwise all equal
