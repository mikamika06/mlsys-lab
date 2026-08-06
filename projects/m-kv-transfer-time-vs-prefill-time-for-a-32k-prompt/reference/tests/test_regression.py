import sys

sys.path.insert(0, ".")
from disagg.transfer import analyze_kv_transfer, kv_cache_bytes
from disagg.sizing import compute_pd_ratio


def test_kv_cache_bytes_includes_k_and_v():
    b = kv_cache_bytes(32768, 32, 8, 128, 2)
    assert b == 4294967296


def test_pd_ratio_accounts_for_transfer_latency():
    r_with = compute_pd_ratio(100.0, 50.0, 1.0, 100)
    r_without = compute_pd_ratio(100.0, 0.0, 1.0, 100)
    assert r_with > r_without
    assert abs(r_with - 1.5) < 1e-6
