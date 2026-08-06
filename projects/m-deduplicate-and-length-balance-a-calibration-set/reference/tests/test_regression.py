from calib.balance import balance_lengths
from calib.dedup import deduplicate_samples


def test_deduplication_exact_and_near():
    s1 = [10, 20, 30, 40, 50, 60, 70, 80]
    s2 = [10, 20, 30, 40, 50, 60, 70, 80]
    s3 = [10, 20, 30, 40, 50, 60, 70, 99]
    s4 = [100, 200, 300, 400, 500, 600, 700, 800]

    deduped = deduplicate_samples([s1, s2, s3, s4], num_perm=128, threshold=0.7)
    assert len(deduped) < 4
    assert s1 in deduped
    assert s4 in deduped


def test_length_balancing():
    samples = [[1] * 5, [1] * 8, [1] * 15, [1] * 25]
    buckets = [10, 20, 30]
    targets = {10: 1, 20: 1, 30: 1}
    balanced = balance_lengths(samples, buckets, targets)
    assert len(balanced) == 3
