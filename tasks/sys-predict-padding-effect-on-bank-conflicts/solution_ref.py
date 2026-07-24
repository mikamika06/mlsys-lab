def choose_padding(width: int) -> int:
    """
    Smallest pad >= 0 such that width+pad is odd (coprime with the 32
    banks), which makes the column-stride warp access fully conflict-free.
    """
    return 0 if width % 2 == 1 else 1
