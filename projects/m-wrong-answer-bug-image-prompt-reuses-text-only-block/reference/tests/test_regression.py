import sys
sys.path.insert(0, ".")
from cache.hash import compute_block_hash, search_collision
from cache.quota import TenantQuotaSimulator


def test_compute_block_hash_distinguishes_modalities():
    data = b"shared_token_payload"
    h_txt = compute_block_hash(data, is_image=False, truncate_bits=32)
    h_img = compute_block_hash(data, is_image=True, truncate_bits=32)
    assert h_txt != h_img


def test_tenant_quota_isolation():
    sim = TenantQuotaSimulator(total_quota=1)
    res1 = sim.allocate("tenant_a", 12345)
    assert res1 is True
    res2 = sim.allocate("tenant_b", 12345)
    assert res2 is False


def test_quota_cleanup():
    sim = TenantQuotaSimulator(total_quota=1)
    sim.allocate("tenant_a", 999)
    sim.free("tenant_a", 999)
    res = sim.allocate("tenant_b", 888)
    assert res is True
