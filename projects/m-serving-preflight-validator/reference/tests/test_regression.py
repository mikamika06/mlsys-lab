"""Regression tests for preflight validator."""

import sys
sys.path.insert(0, ".")
from preflight.validator import validate_fit


def test_validator_blocks_oom():
    """Verify preflight validator rejects configurations exceeding GPU memory."""
    model_config = {
        "num_parameters": 70_000_000_000,
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "max_model_len": 4096,
        "max_num_seqs": 32,
    }
    quant_config = {"quant_type": "fp16"}
    gpu_specs = {
        "num_gpus": 1,
        "memory_per_gpu_bytes": 24 * 1024 * 1024 * 1024,
        "gpu_memory_utilization": 0.90,
    }

    res = validate_fit(model_config, quant_config, gpu_specs)
    assert not res["fits"], "Preflight validator failed to block OOM configuration"
    assert res["headroom_bytes"] < 0, "Headroom should be negative on OOM"
