from coldcache.protocol import ColdCacheProtocol
from coldcache.verifier import verify_cold_execution


def test_cold_cache_verification():
    protocol = ColdCacheProtocol(memory_size=1024)
    requests = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    res = verify_cold_execution(protocol, requests)
    assert res["hits"] == 0
    assert res["strictly_cold"] is True
    assert len(set(res["generations"])) == len(requests)
