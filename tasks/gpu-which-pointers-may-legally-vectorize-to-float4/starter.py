def vectorizable_pointers(addrs: list) -> list:
    """Return True for each address that is 16-byte aligned (legal for float4).

    A float4 load/store requires the byte address to satisfy addr % 16 == 0.
    Return a list of bools, one per address in addrs.
    """
    raise NotImplementedError("your code here")
