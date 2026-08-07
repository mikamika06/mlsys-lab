import math

FP4_MAX = 6.0


def mxfp4_block_exponent(x: list[float], block_size: int = 32) -> list[int]:
    n = len(x)
    nb = n // block_size

    out = [0] * nb

    for i in range(nb):
        start = i * block_size
        amax = 0.0
        for j in range(block_size):
            val = x[start + j]
            if val < 0.0:
                val = -val
            if val > amax:
                amax = val

        if amax == 0.0:
            exp = 0.0
        else:
            exp = math.floor(math.log2(amax / FP4_MAX))

        out[i] = int(exp)

    return out
