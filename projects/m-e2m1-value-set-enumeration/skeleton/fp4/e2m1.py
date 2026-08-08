def decode_e2m1_bits(code: int) -> float:
    """Decode a 4-bit E2M1 integer code (0-15) to its float value."""
    raise NotImplementedError


def enumerate_e2m1() -> list[tuple[int, float]]:
    """Return all 16 (bit_code, decoded_float) pairs in ascending code order."""
    raise NotImplementedError


def quantize_to_e2m1(val: float) -> int:
    """Quantize a float32 value to the nearest 4-bit E2M1 code (0-15)."""
    raise NotImplementedError
