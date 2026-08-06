def reconstruct_tag(rank: int, isa: str) -> str:
    if rank == 4:
        if isa == "avx512":
            return "Acdb16a"
        elif isa == "avx2":
            return "acdb8a"
        else:
            return "abcd"
    elif rank == 2:
        return "ab"
    elif rank == 3:
        return "abc"
    else:
        chars = [chr(ord('a') + i) for i in range(rank)]
        return "".join(chars)
