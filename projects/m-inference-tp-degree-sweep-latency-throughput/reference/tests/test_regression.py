import sys
sys.path.insert(0, ".")
from inference.tp_sweep import sweep_tp_performance, max_valid_tp_degree, verify_server_log_partitions

def test_max_valid_tp_divisibility():
    assert max_valid_tp_degree(8, 32) == 8
    assert max_valid_tp_degree(4, 32) == 4
    assert max_valid_tp_degree(3, 32) == 1

def test_sweep_filtering():
    cfg = {"num_attention_heads": 32, "hidden_dim": 4096, "base_seq_latency": 100.0, "comm_overhead": 2.0, "base_throughput": 1000.0}
    res = sweep_tp_performance(cfg, [1, 2, 3, 4, 8])
    tps = [r["tp"] for r in res]
    assert 3 not in tps
    assert 4 in tps

def test_log_verification():
    logs = [
        "INFO: TensorParallel layer q_proj shape=(4096, 4096)",
        "INFO: TensorParallel layer k_proj shape=(1024, 4096)"
    ]
    assert verify_server_log_partitions(logs, 4) >= 1
