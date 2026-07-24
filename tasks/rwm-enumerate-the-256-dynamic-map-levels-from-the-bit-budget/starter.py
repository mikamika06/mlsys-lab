import numpy as np


def create_dynamic_map(signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8) -> np.ndarray:
    """Enumerate the 256 representable levels of the 8-bit dynamic map.

    The dynamic map is a nonlinear codebook where the total bit budget is
    split between a unary-style exponent field and a fraction field whose
    width SHRINKS as the exponent grows -- so values near zero get a fine
    fraction grid and values near the max magnitude get a coarse one.

    signed: reserve 1 bit for sign (True for this task).
    max_exponent_bits: number of possible exponent field widths, 0..6
        (7 values -> "max_exponent_bits=7").
    total_bits: total bit budget (8 -> 256 levels).

    For each exponent index i in 0..max_exponent_bits-1:
      - non_sign_bits = total_bits - 1  (= 7)
      - fraction_items = 2**i + 1
      - boundaries = linspace(0.1, 1.0, fraction_items)
      - means = midpoints of consecutive boundaries (2**i of them)
      - scale = 10 ** (-(max_exponent_bits - 1) + i)
      - append scale * means (positive branch) and -scale * means
        (negative branch, since signed=True) to the level list

    After the loop, append the two boundary values 0.0 and 1.0.

    Sort the full list ascending and return it as a 1-D float64 NumPy
    array of length `2 ** total_bits` (256 for the defaults).
    """
    raise NotImplementedError('your code here')
