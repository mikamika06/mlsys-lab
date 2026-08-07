import sys

sys.path.insert(0, ".")
from memtrack.stats import parse_vm_stat
from memtrack.zerocopy import verify_zero_copy
from memtrack.compare import compare_copy_costs
import numpy as np


def test_parse_vm_stat_non_negative():
    sample = "Pages wired down: 100.\nPages active: 200.\nPages inactive: 300."
    res = parse_vm_stat(sample)
    for k, v in res.items():
        assert v >= 0


def test_compare_copy_costs_relation():
    res = compare_copy_costs(1024)
    assert res["explicit_cost"] > res["zero_copy_cost"]


def test_verify_zero_copy_types():
    a = np.zeros(10, dtype=np.float32)
    assert isinstance(verify_zero_copy(a, a), bool)
