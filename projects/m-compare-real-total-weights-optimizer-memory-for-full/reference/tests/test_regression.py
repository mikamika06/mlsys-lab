from optmem.compare import compare_memory
from optmem.params import count_parameters
from optmem.states import compute_optimizer_bytes


def test_parameter_counts():
    cfg = {"num_layers": 2, "hidden_dim": 512, "lora_rank": 8, "precision_bytes": 2}
    p = count_parameters(cfg)
    assert p["total_base"] > 0
    assert p["lora_trainable"] < p["full_trainable"]


def test_optimizer_bytes_positive():
    cfg = {"num_layers": 2, "hidden_dim": 512, "lora_rank": 8, "precision_bytes": 2}
    res_full = compute_optimizer_bytes(cfg, "full")
    res_lora = compute_optimizer_bytes(cfg, "lora")
    assert res_full["total"] > res_lora["total"]
    assert res_full["optimizer"] > 0
    assert res_lora["optimizer"] > 0


def test_memory_comparison_ratio():
    cfg = {"num_layers": 2, "hidden_dim": 512, "lora_rank": 8, "precision_bytes": 2}
    comp = compare_memory(cfg)
    assert 0.0 < comp["ratio"] < 1.0
    assert comp["absolute_savings"] > 0
