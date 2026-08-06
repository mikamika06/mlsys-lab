def derive_block_size(isa_name: str, channels: int) -> int:
    if isa_name == "avx512":
        block = 16
    elif isa_name == "avx2":
        block = 8
    elif isa_name == "neon":
        block = 4
    else:
        block = 1
    while channels % block != 0 and block > 1:
        block //= 2
    return block
