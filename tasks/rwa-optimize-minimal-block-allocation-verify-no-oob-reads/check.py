def _oracle(positions, block_size):
    blocks = sorted({p // block_size for p in positions})
    offsets = [p % block_size for p in positions]
    return blocks, offsets


def _valid_gather(block_ids, positions, offsets, block_size):
    block_set = set(block_ids)
    for p, off in zip(positions, offsets):
        if p // block_size not in block_set:
            return False
        if off < 0 or off >= block_size:
            return False
    return True


def grade(sol, fx) -> dict:
    cases = [
        ([0, 3, 8, 9, 16], 8),
        ([31, 32, 33, 63, 64], 32),
        ([5, 5, 5, 6], 4),
        ([100, 7, 15, 16, 17, 200], 8),
        (list(range(20)), 6),
    ]

    ok = 1.0
    for positions, block_size in cases:
        ref_blocks, ref_offsets = _oracle(positions, block_size)
        try:
            got_blocks, got_offsets = sol.minimal_block_allocation(
                list(positions), block_size
            )
            got_blocks = list(got_blocks)
            got_offsets = list(got_offsets)
        except Exception:
            ok = 0.0
            break

        if got_blocks != ref_blocks:
            ok = 0.0
            break
        if got_offsets != ref_offsets:
            ok = 0.0
            break
        if len(got_blocks) != len(set(got_blocks)):
            ok = 0.0
            break
        if not _valid_gather(got_blocks, positions, got_offsets, block_size):
            ok = 0.0
            break

    return {"exact_match": ok}
