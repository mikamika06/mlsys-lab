import sys
sys.path.insert(0, ".")
from zero1.formula import compute_zero1_memory
from zero1.toy_zero import partition_optimizer_states
from zero1.parser import parse_deepspeed_log
import torch


def test_formula_scaling():
    m1 = compute_zero1_memory(1000000, 1, 4, "adam")
    m4 = compute_zero1_memory(1000000, 4, 4, "adam")
    assert m4["optimizer_states_bytes"] < m1["optimizer_states_bytes"]


def test_partition_coverage():
    p1 = torch.nn.Parameter(torch.zeros(10))
    p2 = torch.nn.Parameter(torch.zeros(10))
    s0 = partition_optimizer_states([p1, p2], 2, 0)
    s1 = partition_optimizer_states([p1, p2], 2, 1)
    assert len(s0) > 0
    assert len(s1) > 0


def test_parser_extraction():
    log = "DeepSpeed world_size: 8, Optimizer Params: 5,000,000"
    parsed = parse_deepspeed_log(log)
    assert parsed["world_size"] == 8
    assert parsed["num_params"] == 5000000
