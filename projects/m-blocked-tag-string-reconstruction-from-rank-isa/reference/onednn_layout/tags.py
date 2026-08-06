def reconstruct_tag(rank: int, isa: str, block_size: int = 16) -> str:
    if rank == 4:
        if "avx512" in isa.lower():
            return f"aBcd{block_size}b"
        elif "avx2" in isa.lower():
            return f"aBcd8b"
        else:
            return "nchw"
    elif rank == 5:
        if "avx512" in isa.lower():
            return f"aBcde{block_size}b"
        else:
            return "ncdhw"
    return "abcd"
