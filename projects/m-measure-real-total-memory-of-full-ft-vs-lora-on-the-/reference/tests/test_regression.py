import sys

sys.path.insert(0, ".")
from ftmem.lora import count_lora_params, count_trainable_params
from ftmem.memory import estimate_memory_footprint

CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "vocab_size": 16000,
}

LORA_CONFIG = {"r": 8, "target_modules": ["q_proj", "v_proj"]}


def test_lora_and_qlora_trainable_params_are_identical():
    lora_mem = estimate_memory_footprint(
        CONFIG, mode="lora_bf16", lora_config=LORA_CONFIG
    )
    qlora_mem = estimate_memory_footprint(
        CONFIG, mode="qlora_4bit", lora_config=LORA_CONFIG
    )
    assert (
        lora_mem["trainable_params"] == qlora_mem["trainable_params"]
    ), f"Trainable params mismatch: LoRA={lora_mem['trainable_params']} vs QLoRA={qlora_mem['trainable_params']}"


def test_trainable_params_do_not_include_base_weights():
    lora_p = count_lora_params(CONFIG, LORA_CONFIG)
    trainable_p = count_trainable_params(CONFIG, LORA_CONFIG)
    assert trainable_p == lora_p, f"{trainable_p} != {lora_p}"
    qlora_mem = estimate_memory_footprint(
        CONFIG, mode="qlora_4bit", lora_config=LORA_CONFIG
    )
    assert qlora_mem["trainable_params"] == lora_p


def test_qlora_reduces_base_memory_compared_to_lora_bf16():
    lora_mem = estimate_memory_footprint(
        CONFIG, mode="lora_bf16", lora_config=LORA_CONFIG
    )
    qlora_mem = estimate_memory_footprint(
        CONFIG, mode="qlora_4bit", lora_config=LORA_CONFIG
    )
    assert qlora_mem["base_weights_bytes"] < lora_mem["base_weights_bytes"]
    assert qlora_mem["total_static_bytes"] < lora_mem["total_static_bytes"]
