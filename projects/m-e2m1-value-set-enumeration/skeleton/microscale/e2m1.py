def decode_e2m1(val: int, bias: int, has_nan: bool, has_inf: bool) -> float:
    """Decodes a 4-bit integer (0-15) into a float based on E2M1 rules."""
    raise NotImplementedError


def enumerate_values(bias: int, has_nan: bool, has_inf: bool) -> list[float]:
    """Returns a list of 16 floats representing the decoded values of integers 0 through 15."""
    raise NotImplementedError


def quantize(tensor: list[float], bias: int, has_nan: bool, has_inf: bool) -> list[float]:
    """Quantizes a list of floats to the nearest finite representable E2M1 value."""
    raise NotImplementedError
