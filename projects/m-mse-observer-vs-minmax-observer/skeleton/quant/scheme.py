class QuantizationArgs:
    def __init__(self, bits: int, symmetric: bool, granularity: str):
        raise NotImplementedError


def parse_scheme(name: str) -> QuantizationArgs:
    raise NotImplementedError
