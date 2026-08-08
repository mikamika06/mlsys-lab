"""Reference regression test suite."""

from roofline.flops import compute_prefill_flops_per_token
from roofline.memory import compute_decode_bytes_per_step
from roofline.predictor import predict_decode_throughput

CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008
}


def test_flops_calculation():
    flops_1k = compute_prefill_flops_per_token(CONFIG, 1024)
    flops_2k = compute_prefill_flops_per_token(CONFIG, 2048)
    assert flops_2k > flops_1k


def test_gqa_vs_mha_flops():
    mha_config = dict(CONFIG)
    mha_config["num_key_value_heads"] = 32
    gqa_flops = compute_prefill_flops_per_token(CONFIG, 1024)
    mha_flops = compute_prefill_flops_per_token(mha_config, 1024)
    assert mha_flops > gqa_flops


def test_memory_scaling():
    b1 = compute_decode_bytes_per_step(CONFIG, batch_size=1, context_len=1024)
    b2 = compute_decode_bytes_per_step(CONFIG, batch_size=2, context_len=1024)
    assert b2 > b1


def test_predictor_bounds():
    res = predict_decode_throughput(CONFIG, batch_size=1, context_len=1024, peak_tflops=312.0, hbm_bw_gbps=2000.0)
    assert res["bound"] in ("compute", "memory")
    assert res["tokens_per_sec"] > 0
