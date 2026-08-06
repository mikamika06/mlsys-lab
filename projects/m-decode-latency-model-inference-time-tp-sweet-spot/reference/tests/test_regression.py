from autotp.search import find_tp_sweet_spot
from autotp.model import estimate_decode_latency

def test_partition_invariant():
    config = {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
    hw = {"name": "a100", "memory_bw_gbps": 1555.0, "compute_tflops": 312.0, "comm_bw_gbps": 300.0}
    tp = find_tp_sweet_spot(config, hw, 16, max_tp=8)
    assert tp in [1, 2, 4, 8]
    lat = estimate_decode_latency(config, hw, tp, 16)
    assert lat > 0.0
    assert tp >= 2
