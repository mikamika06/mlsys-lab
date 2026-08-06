import sys
sys.path.insert(0, ".")
from ortopt.audit import audit_decoder_graph
from ortopt.blocklist import get_minimal_blocklist
from ortopt.optimizer import optimize_graph

def test_audit_rejects_missing_kv():
    bad_graph = {"has_kv_cache": False, "valid_precision": True}
    assert not audit_decoder_graph(bad_graph)

def test_blocklist_non_empty():
    bl = get_minimal_blocklist("llama")
    assert len(bl) > 0
    assert "LayerNorm" in bl

def test_optimizer_mode():
    res = optimize_graph("test", "transformers")
    assert "_opt_trans" in res
