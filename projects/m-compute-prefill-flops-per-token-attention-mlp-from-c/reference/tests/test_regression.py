from roofline.analyzer import roofline_tokens_per_sec


def test_roofline_positive_throughput():
    cfg = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 256
    }
    res = roofline_tokens_per_sec(cfg, batch_size=1, context_len=10, hbm_bandwidth_gbps=900.0, tflops=100.0)
    assert res > 0.0
