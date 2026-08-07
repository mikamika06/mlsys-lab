import sys
sys.path.insert(0, ".")

from kvtransfer.config import validate_pair
from kvtransfer.triage import diagnose_stuck_handshake
from kvtransfer.model import estimate_pipelined_transfer_time

def test_transport_type_mismatch_detected():
    p = {
        "role": "producer",
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "dtype": "float16",
        "transport": {"type": "rdma", "gid": "fe80::1", "qp_num": 100}
    }
    c = {
        "role": "consumer",
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "dtype": "float16",
        "transport": {"type": "tcp", "gid": "fe80::1", "qp_num": 100}
    }
    res = validate_pair(p, c)
    assert not res["valid"]
    assert "mismatched transport type" in res["errors"]

def test_stuck_handshake_triage():
    prod_logs = [{"event": "INIT", "session_id": "sess-1"}]
    cons_logs = [{"event": "INIT", "session_id": "sess-2"}]
    diag = diagnose_stuck_handshake(prod_logs, cons_logs)
    assert diag["stuck"] is True
    assert diag["reason"] == "SESSION_ID_MISMATCH"

def test_pipelined_model_speedup():
    m = {"num_layers": 16, "layer_compute_ms": 2.0, "layer_kv_bytes": 1000000}
    n = {"bandwidth_gbps": 10.0, "latency_ms": 0.1}
    res = estimate_pipelined_transfer_time(m, n)
    assert res["speedup"] > 1.0
    assert res["pipelined_ms"] < res["sequential_ms"]
