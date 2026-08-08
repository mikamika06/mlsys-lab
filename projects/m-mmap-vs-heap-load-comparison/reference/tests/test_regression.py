import sys

sys.path.insert(0, ".")
from memload.attribution import attribute_size_regression
from memload.dedup import calculate_dedup_savings
from memload.loader import compare_load_footprint


def test_mmap_resident_bytes_is_page_multiple():
    tensors = [
        {"name": "t1", "offset": 100, "size_bytes": 1000, "accessed_bytes": 500},
        {"name": "t2", "offset": 4500, "size_bytes": 2000, "accessed_bytes": 1000},
    ]
    res = compare_load_footprint(tensors, page_size=4096)
    assert res["mmap_resident_bytes"] % 4096 == 0
    assert res["mmap_resident_bytes"] >= sum(t["accessed_bytes"] for t in tensors)


def test_dedup_savings_never_exceeds_raw_total():
    tensors = [
        {"name": "t1", "hash": "h1", "offset": 0, "size_bytes": 1000, "accessed_bytes": 1000},
        {"name": "t2", "hash": "h1", "offset": 1000, "size_bytes": 1000, "accessed_bytes": 1000},
    ]
    res = calculate_dedup_savings(tensors, page_size=4096)
    assert res["disk_savings_bytes"] <= res["raw_total_bytes"]
    assert res["disk_savings_bytes"] == 1000


def test_attribution_category_deltas_sum_to_net_delta():
    base = [{"name": "t1", "layer": "l0", "size_bytes": 1000}]
    cand = [
        {"name": "t1", "layer": "l0", "size_bytes": 1200},
        {"name": "t2", "layer": "l1", "size_bytes": 500},
    ]
    res = attribute_size_regression(base, cand)
    cats = res["category_deltas"]
    total_cat = cats["added"] + cats["removed"] + cats["modified"]
    assert total_cat == res["net_delta_bytes"]
