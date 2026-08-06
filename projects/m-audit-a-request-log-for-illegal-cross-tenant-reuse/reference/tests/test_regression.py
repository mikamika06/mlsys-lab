from logaudit.audit import parse_and_audit


def test_cross_tenant_isolation():
    logs = [
        {"type": "allocate", "block_id": 101, "tenant_id": "tenant_A", "tokens": [1, 2, 3, 4]},
        {"type": "lookup", "request_id": "req_1", "block_id": 101, "tenant_id": "tenant_A"},
        {"type": "lookup", "request_id": "req_2", "block_id": 101, "tenant_id": "tenant_B"},
    ]
    violations = parse_and_audit(logs)
    assert len(violations) == 1
    assert violations[0]["tenant_id"] == "tenant_B"
    assert violations[0]["owner_tenant_id"] == "tenant_A"
    assert violations[0]["block_id"] == 101
    assert violations[0]["tokens_leaked"] == 4
