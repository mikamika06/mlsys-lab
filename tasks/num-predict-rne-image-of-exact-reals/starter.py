def rne_fp32_bits(pairs: list[tuple[int, int]]) -> list[int]:
    """For each exact rational (num, den) with den > 0, return the uint32
    bit pattern of the float32 it rounds to under round-to-nearest-even.

    Must round directly from the exact integer pair (no intermediate
    float64 cast -- some inputs are constructed to make that double-round
    incorrectly).
    """
    raise NotImplementedError('your code here')
