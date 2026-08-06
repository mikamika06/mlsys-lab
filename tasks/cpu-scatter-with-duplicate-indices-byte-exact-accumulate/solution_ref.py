def scatter_add(dst: list[int], idx: list[int], src: list[int], out: list[int]) -> None:
    """Accumulate src into out at indices idx, handling duplicates."""
    for i in range(len(src)):
        out[idx[i]] += src[i]
