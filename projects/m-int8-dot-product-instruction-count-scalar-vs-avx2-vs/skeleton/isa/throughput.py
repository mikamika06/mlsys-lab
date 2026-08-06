def analyze_throughput(isa: str, vec_bits: int, ports: int) -> dict:
    """
    Returns a dictionary with:
    - "elements_per_vec": number of int8 elements processed per sequence
    - "instructions_per_sequence": number of instructions in the sequence
    - "macs_per_cycle": float representing MACs per cycle (rounded to 2 decimals)
    """
    raise NotImplementedError
