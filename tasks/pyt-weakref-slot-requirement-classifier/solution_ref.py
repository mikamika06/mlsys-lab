import weakref

def can_weakref(o: object) -> bool:
    """
    Return True if ``weakref.ref(o)`` succeeds, otherwise False.
    """
    try:
        weakref.ref(o)
        return True
    except TypeError:
        return False
