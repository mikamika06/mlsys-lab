def snake(name):
    """`ToolCalls` -> `tool_calls`: the Go field name as its JSON key."""
    raise NotImplementedError


def is_true(v):
    """Go's template truth: zero number, empty string, empty collection, nil
    and false are all false. Everything else is true."""
    raise NotImplementedError


def to_text(v):
    """A value as the template would print it.

    true/false print as `true`/`false`. A value whose type defines its own
    __str__ prints through it — that is how a Go type with a String method
    behaves, and two values in the fixtures rely on it. A key the data does not
    carry is a zero-valued field, not a nil interface.
    """
    raise NotImplementedError
