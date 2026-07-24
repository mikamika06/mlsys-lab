import numpy as np


def _oracle(seqlens, n_blocks, block_size, max_len):
    used = 0
    paged = 0
    for length in seqlens:
        blocks = -(-int(length) // block_size)  # ceil division
        if used + blocks > n_blocks:
            break
        used += blocks
        paged += 1

    contig_blocks_per_req = -(-int(max_len) // block_size)
    contig = n_blocks // contig_blocks_per_req

    return paged, contig


def grade(sol, fx) -> dict:
    seqlens = np.asarray(fx["seqlens"], dtype=np.int64)

    configs = [
        (100, 16, 512),
        (50, 32, 512),
        (200, 16, 512),
        (30, 16, 256),
        (80, 8, 512),
    ]

    ok = 1.0
    for n_blocks, block_size, max_len in configs:
        expected = _oracle(seqlens, n_blocks, block_size, max_len)
        try:
            got = sol.paged_vs_contiguous_concurrency(
                seqlens.copy(), n_blocks, block_size, max_len
            )
            got = tuple(int(x) for x in got)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
