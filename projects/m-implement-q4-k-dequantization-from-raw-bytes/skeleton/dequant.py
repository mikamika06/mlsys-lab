def unpack_6bit_scales_and_mins(q: bytes) -> tuple[list[int], list[int]]:
    raise NotImplementedError


def dequantize_q4_k(block: bytes) -> list[float]:
    raise NotImplementedError


def dequantize_q6_k(block: bytes) -> list[float]:
    raise NotImplementedError
