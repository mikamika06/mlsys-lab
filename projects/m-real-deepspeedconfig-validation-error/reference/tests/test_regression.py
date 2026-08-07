import sys

sys.path.insert(0, ".")
from dsdiag.bucket import optimal_bucket_size


def test_bucket_size_respects_ceiling():
    sizes = [100, 500, 1000, 5000]
    ceiling = 1200
    res = optimal_bucket_size(sizes, ceiling)
    assert res <= ceiling, f"bucket size {res} exceeds ceiling {ceiling}"
    assert res == 1000
