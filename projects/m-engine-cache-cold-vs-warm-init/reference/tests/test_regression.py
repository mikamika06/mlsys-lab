import sys
sys.path.insert(0, ".")
from trtcache.engine import classify_init_state, verify_cache_validity
from trtcache.partition import compute_node_coverage, filter_subgraphs
from trtcache.monitor import evaluate_latency_ratio


def test_classify_init_state_cold():
    meta = {"hash": "abc", "profile_signature": "p1", "plugin_version": "1.0"}
    store = {}
    assert classify_init_state(meta, store) == "cold"


def test_classify_init_state_warm():
    meta = {"hash": "abc", "profile_signature": "p1", "plugin_version": "1.0"}
    store = {"abc": {"profile_signature": "p1", "plugin_version": "1.0"}}
    assert classify_init_state(meta, store) == "warm"


def test_verify_cache_validity_mismatch():
    meta = {"profile_signature": "p1", "is_valid": True}
    assert verify_cache_validity(meta, "p2") is False


def test_compute_node_coverage_full():
    nodes = ["n1", "n2", "n3"]
    sgs = [{"nodes": ["n1", "n2", "n3"]}]
    assert compute_node_coverage(nodes, sgs) == 1.0


def test_evaluate_latency_ratio():
    cold = [100.0, 100.0]
    warm = [10.0, 10.0]
    assert evaluate_latency_ratio(cold, warm) == 0.1
