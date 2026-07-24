import numpy as np

def decode_e4m3(codes):
    """Decode an array of E4M3 uint8 bit patterns to float64 values.

    E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
    No infinity or NaN special values — every code is a finite number.
    """
    raise NotImplementedError("your code here")
