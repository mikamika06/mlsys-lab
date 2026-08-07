import sys
sys.path.insert(0, ".")
from trtplug.audit import audit_bindings

def test_audit_catches_missing_plugin():
    graph = {
        "nodes": [
            {"name": "custom_op_1", "domain": "custom.domain", "op_type": "FusedAttention"}
        ]
    }
    registered = {}
    result = audit_bindings(graph, registered)
    assert not result["is_fully_bound"]
    assert "custom_op_1" in result["unbound_nodes"]

def test_audit_passes_when_bound():
    graph = {
        "nodes": [
            {"name": "custom_op_1", "domain": "custom.domain", "op_type": "FusedAttention"}
        ]
    }
    registered = {"custom.domain::FusedAttention": True}
    result = audit_bindings(graph, registered)
    assert result["is_fully_bound"]
    assert len(result["unbound_nodes"]) == 0
