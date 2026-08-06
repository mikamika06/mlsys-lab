import sys
sys.path.insert(0, ".")
from ring.bucket import assign_buckets

PARAMS = [
    {"name": "layer1.weight", "size_bytes": 1024 * 1024},
    {"name": "layer2.weight", "size_bytes": 1024 * 1024},
    {"name": "layer3.weight", "size_bytes": 1024 * 1024},
]

def test_reverse_order_assignment():
    buckets = assign_buckets(PARAMS, bucket_cap_mb=2)
    first_bucket_names = buckets[0]
    assert "layer3.weight" in first_bucket_names, "Buckets must be assigned in reverse order of parameters"

def test_bucket_non_empty():
    buckets = assign_buckets(PARAMS, bucket_cap_mb=10)
    assert len(buckets) > 0
    for b in buckets:
        assert len(b) > 0
