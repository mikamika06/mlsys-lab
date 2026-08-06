import sys

sys.path.insert(0, ".")
from trt_engine.cache import verify_engine_cache, compute_warm_init_latency
from trt_engine.partition import compute_node_coverage
from trt_engine.validator import validate_cache_fingerprint, detect_invalidation_trigger


def test_cache_verification_matches_valid_config():
    meta = {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 64, 64], "max": [1, 3, 256, 256]}]}
    cfg = {"device_id": 0, "trt_version": "8.6.1", "profiles": [{"min": [1, 3, 64, 64], "max": [1, 3, 256, 256]}]}
    assert verify_engine_cache(meta, cfg) is True


def test_node_coverage_full():
    nodes = ["Conv_1", "Relu_2", "Gemm_3"]
    subs = [{"nodes": ["Conv_1", "Relu_2", "Gemm_3"]}]
    assert compute_node_coverage(nodes, subs) == 1.0


def test_invalidator_detects_mismatch():
    p1 = {"min": [1, 3, 32, 32], "opt": [1, 3, 64, 64], "max": [1, 3, 128, 128]}
    p2 = {"min": [1, 3, 64, 64], "opt": [1, 3, 64, 64], "max": [1, 3, 128, 128]}
    assert detect_invalidation_trigger(p1, p2) is True
