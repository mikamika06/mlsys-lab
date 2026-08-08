def decode_e2m1_bits(code: int) -> float:
    code = code & 0x0F
    sign = (code >> 3) & 0x01
    exp = (code >> 1) & 0x03
    mant = code & 0x01

    if exp == 0:
        val = (2.0 ** (-1)) * (0.0 + mant * 0.5)
    else:
        bias = 1
        val = (2.0 ** (exp - bias)) * (1.0 + mant * 0.5)

    return -val if sign else val


def enumerate_e2m1() -> list[tuple[int, float]]:
    return [(i, decode_e2m1_bits(i)) for i in range(16)]


def quantize_to_e2m1(val: float) -> int:
    candidates = enumerate_e2m1()

    best_code = 0
    best_diff = float("inf")

    for code, candidate_val in candidates:
        diff = abs(val - candidate_val)
        if diff < best_diff:
            best_diff = diff
            best_code = code
        elif diff == best_diff:
            if (code & 1) == 0:
                best_code = code

    return best_code
