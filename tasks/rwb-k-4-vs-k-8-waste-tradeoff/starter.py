def compare_k4_k8_waste(sizes: list[int], counts: list[int]) -> tuple[int, int, int]:
    """Compare optimal padding waste at K=4 vs K=8 buckets.

    sizes: 1-D array of distinct observed request sizes.
    counts: 1-D array, same length as `sizes`, counts[i] = how many
        requests of size sizes[i] were observed.

    For each K in {4, 8}: sort the distinct sizes ascending, and find the
    K contiguous ranges (each range covered by a single bucket equal to
    its largest size) that minimize

        total_waste = sum_i counts[i] * (bucket(sizes[i]) - sizes[i])

    where bucket(s) is the smallest chosen bucket >= s -- i.e. run the
    optimal bucket-selection DP (see rwb-dp-optimal-bucket-selection-k-buckets)
    once for k=4 and once for k=8. If there are fewer than K distinct
    sizes, using all of them (zero waste) is optimal.

    Returns (waste_k4, waste_k8, reduction) where
    reduction = waste_k4 - waste_k8 (always >= 0: more buckets can never
    increase the optimal achievable waste).
    """
    raise NotImplementedError('your code here')
