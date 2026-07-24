def process(sizes: list, make_buffer) -> float:
    """Sum make_buffer(s).checksum() over every s in sizes, without ever
    holding more than the minimum necessary number of large objects
    alive at once (stream, don't collect-then-process).
    """
    raise NotImplementedError('your code here')
