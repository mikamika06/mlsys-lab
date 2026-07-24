def lanes_per_register(reg_bits: int) -> dict:
    """
    Return the number of float32, float16 and int8 values that fit in a SIMD register
    of width reg_bits (in bits).  Raises ValueError if reg_bits is not positive.
    """
    if reg_bits <= 0:
        raise ValueError("reg_bits must be a positive integer")
    return {
        "float32": reg_bits // 32,
        "float16": reg_bits // 16,
        "int8":   reg_bits // 8,
    }
