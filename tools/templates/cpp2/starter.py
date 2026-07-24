def struct_size(fields):
    """Return sizeof of a C struct with these field types (in order) under the
    pinned LP64 ABI: natural alignment + inter-field padding + tail padding.
    fields e.g. ['char','int','double']."""
    raise NotImplementedError('your code here')
