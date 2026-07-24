def vectorizable_pointers(addrs: list) -> list:
    """Return True for each address that is 16-byte aligned (legal for float4).

    addr % 16 == 0 is the alignment requirement for float4 loads/stores.
    """
    return [addr % 16 == 0 for addr in addrs]
