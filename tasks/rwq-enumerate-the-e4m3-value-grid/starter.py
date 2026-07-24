import numpy as np


def e4m3_value_grid() -> dict:
    """Enumerate all 256 E4M3 (S,E,M) codes, decode, drop the 2 NaN codes,
    and return {"values": sorted unique finite float64 array,
    "n_finite": int, "max_finite": float, "min_subnormal": float}."""
    raise NotImplementedError('your code here')
