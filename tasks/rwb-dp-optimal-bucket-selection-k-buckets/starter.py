def select_buckets(size_histogram, k):
    """Choose up to k bucket sizes minimizing total padding waste.

    size_histogram: {size: count}, positive ints.
    k: bucket budget (positive int).

    Every observed size rounds up to the smallest chosen bucket >= it;
    total_waste = sum(count(s) * (bucket(s) - s)). The largest observed
    size must be covered by a bucket. If k >= number of distinct sizes,
    using every size as its own bucket (zero waste) is optimal.

    Returns (buckets, total_waste): buckets is a list of chosen sizes
    (each drawn from size_histogram's keys) achieving the minimum
    possible total_waste; total_waste is the actual waste they produce.
    """
    raise NotImplementedError('your code here')
