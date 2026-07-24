def zero_copy_slice(buf: bytearray, start: int, stop: int) -> memoryview:
    return memoryview(buf)[start:stop]
