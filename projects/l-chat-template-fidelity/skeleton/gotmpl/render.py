def render(src, data, funcs=None):
    """Render a Go text/template source against `data`.

    `funcs` adds to the built-ins (eq ne lt le gt ge not len index and or).
    Ranging over a mapping visits its keys in sorted order, which is what Go
    does and what the recorded output depends on.
    """
    raise NotImplementedError
