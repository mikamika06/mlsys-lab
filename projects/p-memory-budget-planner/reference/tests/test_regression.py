import sys
sys.path.insert(0, ".")
from planner.calculator import MemoryPlanner

def test_zero3_reduces_weights():
    cfg = {"zero_stage": 3, "world_size": 4, "num_params": 1000, "bytes_per_param": 2}
    p = MemoryPlanner(cfg)
    cfg_base = {"zero_stage": 0, "world_size": 4, "num_params": 1000, "bytes_per_param": 2}
    p_base = MemoryPlanner(cfg_base)
    assert p.weights_memory() < p_base.weights_memory()

def test_checkpointing_reduces_memory():
    cfg_off = {"activation_checkpointing": False, "num_layers": 10, "micro_batch_size": 1, "seq_len": 100, "hidden_size": 100}
    cfg_on = {"activation_checkpointing": True, "num_layers": 10, "micro_batch_size": 1, "seq_len": 100, "hidden_size": 100}
    p_off = MemoryPlanner(cfg_off)
    p_on = MemoryPlanner(cfg_on)
    assert p_on.activations_memory() < p_off.activations_memory()

def test_advise_emits_suggestions():
    cfg = {"zero_stage": 0, "activation_checkpointing": False, "cpu_offload": False, "micro_batch_size": 8}
    p = MemoryPlanner(cfg)
    limit = p.total_memory() // 2
    advice = p.advise(limit)
    assert isinstance(advice, list)
    assert len(advice) > 0
