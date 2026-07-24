from __future__ import annotations

import random

def _ffd_reference(lengths, bin_size):
    """Reference first-fit decreasing bin packing oracle."""
    seq = list(lengths)
    n = len(seq)
    indexed = sorted(enumerate(seq), key=lambda x: x[1], reverse=True)
    bins = []               # remaining capacity per bin
    assignment = [0] * n

    for orig_idx, l in indexed:
        placed = False
        for b_idx, cap in enumerate(bins):
            if cap >= l:
                bins[b_idx] = cap - l
                assignment[orig_idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append(bin_size - l)
            assignment[orig_idx] = len(bins) - 1

    return len(bins), assignment

def grade(sol, fx):
    """Grade the student's pack_into_fixed_bins against the FFD reference."""
    test_cases = [
        # (lengths, bin_size)
        ([], 10),
        ([5], 10),
        ([10, 10, 10], 15),
        ([4, 2, 3, 5], 5),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1], 12),
        ([7, 7, 7, 7, 7], 20),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15),
        ([3, 3, 3, 3, 3, 3], 10),
        ([8, 1, 8, 1, 8, 1], 10),
        ([1], 1),
    ]

    # Deterministic random fixture
    rng = random.Random(12345)
    rand_lengths = [rng.randint(1, 50) for _ in range(30)]
    rand_bin_size = rng.randint(20, 100)
    test_cases.append((rand_lengths, rand_bin_size))

    for lengths, bs in test_cases:
        # Reference answer
        try:
            ref_num, ref_assign = _ffd_reference(lengths, bs)
        except Exception:
            return {"exact_match": 0.0}

        # Student answer
        try:
            stu_num, stu_assign = sol.pack_into_fixed_bins(lengths, bs)
        except Exception:
            return {"exact_match": 0.0}

        # Compare number of bins
        if ref_num != stu_num:
            return {"exact_match": 0.0}

        # Normalise both to plain Python lists of ints
        ref_list = [int(x) for x in ref_assign]
        stu_list = [int(x) for x in stu_assign]
        if ref_list != stu_list:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
