import sys
sys.path.insert(0, ".")
from checkpoint.policy import select_checkpoint_policy
from checkpoint.dropout import verify_dropout_consistency
from checkpoint.compile_bridge import optimize_min_cut
import numpy as np

def test_policy_within_budget():
    cfg = {"layers": 3, "mem_costs": [10, 20, 30], "compute_costs": [5, 5, 5], "budget": 25}
    policy = select_checkpoint_policy(cfg, cfg["budget"])
    mem_used = sum(cfg["mem_costs"][i] for i in range(cfg["layers"]) if not policy[i])
    assert mem_used <= cfg["budget"]

def test_dropout_consistency_check():
    m = np.ones((2, 2))
    assert verify_dropout_consistency(m, m) is True

def test_compile_min_cut_basic():
    nodes = [10, 20, 30, 40]
    cuts = optimize_min_cut(nodes, 35)
    assert isinstance(cuts, list)
