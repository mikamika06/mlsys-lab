def _ref(sizes, buckets):
    """Correct ceiling assignment: smallest bucket >= s else largest."""
    indices = []
    for s in sizes:
        idx = 0
        while idx < len(buckets) - 1 and buckets[idx] < s:
            idx += 1
        indices.append(idx)
    return indices

def grade(sol, fx) -> dict:
    # Test cases designed to catch the "nearest bucket" bug.
    test_cases = [
        ([5, 7, 12, 30], [8, 12, 20]),       # from example
        ([1, 2, 3, 4, 5], [5]),               # single bucket
        ([10, 20, 30], [5, 15, 25, 35]),      # mixed fit / fallback
        ([7, 8, 9], [3, 10]),                 # nearest goes wrong for 7
        ([6, 11, 17], [5, 10, 15]),           # 6 – nearest = 5 (wrong); 17 – fallback
        ([0], [2, 4]),                        # zero size
        ([2, 2, 2], [3, 5]),                  # duplicates
        ([10, 20, 30, 5, 15, 25], [8, 16, 24]),
        ([1, 100, 2, 200], [50, 150]),        # extreme sizes
    ]

    try:
        for sizes, buckets in test_cases:
            got = sol.bucket_assign(sizes, buckets)
            expected = _ref(sizes, buckets)
            if got != expected:
                return {"exact_match": 0.0}
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
