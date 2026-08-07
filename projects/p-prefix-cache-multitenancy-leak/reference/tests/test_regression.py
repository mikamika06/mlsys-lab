import sys

sys.path.insert(0, ".")
from prefix_cache import PrefixCache


def test_isolation_prevents_leak():
    c = PrefixCache(isolate=True)
    tokens = [1, 2, 3, 4]
    c.insert(tokens, tenant_id="a")
    hits = c.lookup(tokens, tenant_id="b")
    assert hits == 0, f"Expected 0 hits across tenants, got {hits}"


def test_system_prefix_sharing():
    c = PrefixCache(isolate=True)
    sys_pfx = [1, 2]
    tokens = [1, 2, 3, 4]
    c.insert(tokens, tenant_id="a", system_prefixes=[sys_pfx])
    hits = c.lookup(tokens, tenant_id="b", system_prefixes=[sys_pfx])
    assert hits >= 2, f"Expected system prefix hits, got {hits}"
