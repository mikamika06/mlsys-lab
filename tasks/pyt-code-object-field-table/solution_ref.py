def code_fields(fn):
    """Return (co_argcount, co_nlocals, co_stacksize, co_flags,
    len(co_consts), len(co_names)) for fn's code object."""
    c = fn.__code__
    return (
        c.co_argcount,
        c.co_nlocals,
        c.co_stacksize,
        c.co_flags,
        len(c.co_consts),
        len(c.co_names),
    )
