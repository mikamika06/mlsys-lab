import sys
sys.path.insert(0, ".")
from vllm_sec.isolation import compute_block_hashes, check_tenant_isolation


def test_tenant_isolation_no_shared_blocks():
    tokens = [101, 2054, 2003, 1037, 3829, 2000, 1037, 2742] * 4
    block_size = 16

    hashes_tenant_a = compute_block_hashes(tokens, block_size, "salt_tenant_a_9921")
    hashes_tenant_b = compute_block_hashes(tokens, block_size, "salt_tenant_b_4431")

    shared = set(hashes_tenant_a).intersection(set(hashes_tenant_b))
    assert len(shared) == 0, f"Found shared block hashes across salts: {shared}"


def test_isolation_checker_detects_no_leakage():
    reqs = [
        {"tenant_id": "t1", "tenant_salt": "salt1", "tokens": [10, 20, 30, 40] * 8},
        {"tenant_id": "t2", "tenant_salt": "salt2", "tokens": [10, 20, 30, 40] * 8},
    ]
    is_isolated, shared_count = check_tenant_isolation(reqs, block_size=16)
    assert is_isolated is True
    assert shared_count == 0
