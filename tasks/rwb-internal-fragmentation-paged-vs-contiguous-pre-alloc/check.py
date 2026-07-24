import math
import numpy as np

def _ref_paged_waste(seqlens, block_size):
    """Oracle: total paged waste = sum(ceil(l/B)*B - l)."""
    seqlens = np.asarray(seqlens, dtype=np.int64)
    full_blocks = np.ceil(seqlens / float(block_size)).astype(np.int64)
    return int(np.sum(full_blocks * block_size - seqlens))

def _ref_contig_waste(seqlens, max_len):
    """Oracle: total contiguous waste = sum(L - l)."""
    seqlens = np.asarray(seqlens, dtype=np.int64)
    return int(np.sum(max_len - seqlens))

def _rel_err(got, ref):
    return abs(float(got) - float(ref)) / (abs(float(ref)) + 1e-12)

def grade(sol, fx) -> dict:
    cases = [
        # (seqlens, block_size, max_len)
        (np.array([3, 7, 13]), 5, 16),
        (np.array([3, 64, 65, 127, 128, 200, 512, 7, 31, 129]), 64, 512),
        (np.array([1]), 1, 1),
        (np.array([100, 100, 100]), 100, 100),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3, 20),
        (np.array([63, 64, 65]), 64, 128),
        (np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 16, 100),
        (np.arange(1, 257)),  # 1..256, block_size=32, max_len=256
    ]

    # Override last case to have explicit params
    cases[-1] = (np.arange(1, 257), 32, 256)

    max_paged_err = 0.0
    max_contig_err = 0.0

    for seqlens, block_size, max_len in cases:
        try:
            got_paged, got_contig = sol.internal_fragmentation(
                seqlens, block_size, max_len
            )
        except Exception:
            return {"paged_rel_err": 1.0, "contig_rel_err": 1.0}

        ref_paged = _ref_paged_waste(seqlens, block_size)
        ref_contig = _ref_contig_waste(seqlens, max_len)

        pe = _rel_err(got_paged, ref_paged)
        ce = _rel_err(got_contig, ref_contig)

        max_paged_err = max(max_paged_err, pe)
        max_contig_err = max(max_contig_err, ce)

    return {
        "paged_rel_err": max_paged_err,
        "contig_rel_err": max_contig_err,
    }
