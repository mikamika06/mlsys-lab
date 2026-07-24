def class_cell_info(method):
    """Inspect a method's code object for the implicit __class__ closure
    cell that zero-arg super() (and bare __class__) rely on. Returns the
    referenced class's __name__, or None if the method carries no such
    cell."""
    raise NotImplementedError('your code here')
