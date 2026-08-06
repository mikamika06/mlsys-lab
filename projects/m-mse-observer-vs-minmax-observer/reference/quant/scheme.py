class QuantizationArgs:
    def __init__(self, bits: int, symmetric: bool, granularity: str):
        self.bits = bits
        self.symmetric = symmetric
        self.granularity = granularity


def parse_scheme(name: str) -> QuantizationArgs:
    parts = name.split("-")
    bits = int(parts[0].replace("int", ""))
    symmetric = "sym" in parts
    granularity = parts[-1] if parts[-1] in ("tensor", "channel", "block") else "tensor"
    return QuantizationArgs(bits=bits, symmetric=symmetric, granularity=granularity)
