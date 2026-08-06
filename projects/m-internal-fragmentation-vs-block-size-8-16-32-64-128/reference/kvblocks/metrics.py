def compute_fragmentation(seq_lens: list[int], block_sizes: list[int]) -> dict[int, int]:
    res = {}
    for bs in block_sizes:
        frag = sum(((l + bs - 1) // bs) * bs - l for l in seq_lens)
        res[bs] = frag
    return res
