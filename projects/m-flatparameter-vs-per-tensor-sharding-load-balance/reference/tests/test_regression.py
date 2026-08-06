import sys

sys.path.insert(0, ".")
from fsdp_balance.sharding import auto_wrap_assign, check_freeze_constraint, compute_load_balance


def test_load_balance_positive():
    ratio = compute_load_balance([100, 200, 300], 2, "flat")
    assert ratio > 0


def test_auto_wrap_non_empty():
    tree = {"name": "root", "params": 100, "children": []}
    units = auto_wrap_assign(tree, 50)
    assert len(units) > 0


def test_freeze_constraint_raises():
    try:
        check_freeze_constraint(1000, 500)
        assert False, "should have raised"
    except RuntimeError:
        pass
