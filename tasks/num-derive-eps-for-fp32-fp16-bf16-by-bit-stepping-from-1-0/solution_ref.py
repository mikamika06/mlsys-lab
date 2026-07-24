import numpy as np

def derive_eps() -> tuple[float, float, float]:
    eps32 = np.nextafter(np.float32(1.0), np.float32(np.inf)) - np.float32(1.0)
    eps16 = np.nextafter(np.float16(1.0), np.float16(np.inf)) - np.float16(1.0)

    # Compute BF16 epsilon via bit manipulation to avoid dependency on np.bfloat16
    bits = np.uint32(0x3f800000)          # float32 representation of 1.0
    next_bits = bits + (1 << 16)          # increment mantissa in the top 16 bits
    epsbf16 = np.float32(next_bits.view(np.float32)) - 1.0

    return (float(eps32), float(eps16), float(epsbf16))
