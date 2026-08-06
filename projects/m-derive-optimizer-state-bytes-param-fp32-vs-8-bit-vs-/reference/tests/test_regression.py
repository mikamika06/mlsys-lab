from optbudget.bytes import derive_optimizer_bytes_per_param
from optbudget.budget import derive_total_memory_budget
from optbudget.spill import derive_spill_trigger_step


def test_optimizer_bytes_consistency():
    assert derive_optimizer_bytes_per_param("adam_fp32") == 12.0
    assert derive_optimizer_bytes_per_param("adam_8bit") == 2.0
    assert derive_optimizer_bytes_per_param("sgd") == 4.0


def test_memory_budget_positive():
    cfg = {
        "trainable_params": 10000000,
        "total_params": 1000000000,
        "base_precision_bits": 16,
        "lora_precision_bits": 16,
        "activation_bytes": 100,
        "gradient_bytes": 100,
        "optimizer_type": "adam_8bit",
        "vram_limit_bytes": 10000000000,
        "base_spill_step": 10,
        "step_growth_rate": 1.0
    }
    budget = derive_total_memory_budget(cfg)
    assert budget > 0.0


def test_spill_trigger_step_validity():
    cfg = {
        "trainable_params": 100000000,
        "total_params": 7000000000,
        "base_precision_bits": 16,
        "lora_precision_bits": 16,
        "activation_bytes": 1024,
        "gradient_bytes": 1024,
        "optimizer_type": "adam_fp32",
        "vram_limit_bytes": 1000,
        "base_spill_step": 10,
        "step_growth_rate": 1.0
    }
    step = derive_spill_trigger_step(cfg)
    assert isinstance(step, int)
    assert step >= 0
