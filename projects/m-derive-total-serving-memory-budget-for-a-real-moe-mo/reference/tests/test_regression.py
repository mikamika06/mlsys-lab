from moebudget.latency import measure_latency


def test_latency_outputs():
    cfg = {
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
        "num_experts": 8,
        "active_experts": 2,
        "expert_hidden_size": 4096,
        "bytes_per_param": 2,
        "context_len": 4096,
        "batch_size": 4,
    }
    contexts = [1024, 2048]
    res = measure_latency(cfg, contexts)
    assert len(res) == len(contexts)
    for prefill, decode in res:
        assert prefill > 0.0
        assert decode > 0.0
