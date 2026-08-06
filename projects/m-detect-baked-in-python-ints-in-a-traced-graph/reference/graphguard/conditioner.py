def rewrite_conditional(fn):
    """Rewrite conditional construct."""
    def wrapped(x):
        return fn(x)
    return wrapped
