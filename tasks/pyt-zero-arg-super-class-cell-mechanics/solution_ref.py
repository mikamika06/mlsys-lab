def class_cell_info(method):
    """Inspect a method's code object for the implicit __class__ closure
    cell that zero-arg super() (and bare __class__) rely on.

    If the compiler detected a bare `super` or `__class__` reference in the
    method's body, `__class__` appears in `method.__code__.co_freevars` and
    `method.__closure__` holds a cell bound to the lexically enclosing
    class. Returns that class's `__name__`, or None if the method carries
    no such cell.
    """
    freevars = method.__code__.co_freevars
    if "__class__" not in freevars:
        return None
    idx = freevars.index("__class__")
    cell = method.__closure__[idx]
    return cell.cell_contents.__name__
