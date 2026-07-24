def reinterpret_roundtrip(data):
    view = memoryview(data)
    ints = view.cast("i")
    bytes_view = ints.cast("B")
    floats = bytes_view.cast("f")
    return floats.cast("B").tobytes()
