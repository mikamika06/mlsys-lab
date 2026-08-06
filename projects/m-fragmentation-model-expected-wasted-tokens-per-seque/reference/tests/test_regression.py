import sys
sys.path.insert(0, ".")
from frag.model import expected_wasted_tokens, optimal_block_size
from frag.auditor import audit_block_trace


def test_wasted_tokens_non_negative():
    hist = {32: 10, 64: 5}
    for bs in [16, 32]:
        assert expected_wasted_tokens(hist, bs) >= 0.0


def test_optimal_block_size_in_candidates():
    hist = {32: 10, 64: 5}
    cands = [16, 32, 64]
    assert optimal_block_size(hist, cands, 1.0) in cands


def test_auditor_detects_leaks():
    trace = [{"type": "allocate", "seq_id": 1, "block_id": 10}, {"type": "terminate", "seq_id": 1}]
    res = audit_block_trace(trace)
    assert res["leaked"] == 1
