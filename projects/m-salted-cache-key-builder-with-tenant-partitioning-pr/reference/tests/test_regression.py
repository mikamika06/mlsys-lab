import sys

sys.path.insert(0, ".")
from cachekey.builder import build_prefix_keys
from cachekey.oracle import can_share_blocks


def test_different_tenants_produce_different_keys():
    tokens = [101, 202, 303, 404, 505, 606, 707, 808]
    keys_a = build_prefix_keys(tenant_id="tenant_A", tokens=tokens, block_size=4, salt="secret")
    keys_b = build_prefix_keys(tenant_id="tenant_B", tokens=tokens, block_size=4, salt="secret")
    assert keys_a != keys_b, "Different tenants generated identical cache keys"


def test_cannot_share_blocks_across_tenants_by_default():
    req_a = {"tenant_id": "tenant_1", "salt": "s1", "tokens": [1, 2, 3]}
    req_b = {"tenant_id": "tenant_2", "salt": "s1", "tokens": [1, 2, 3]}
    assert not can_share_blocks(req_a, req_b, allow_cross_tenant=False)


def test_salt_isolation_prevents_sharing():
    req_a = {"tenant_id": "tenant_1", "salt": "salt_A", "tokens": [1, 2, 3]}
    req_b = {"tenant_id": "tenant_1", "salt": "salt_B", "tokens": [1, 2, 3]}
    assert not can_share_blocks(req_a, req_b, allow_cross_tenant=True)
