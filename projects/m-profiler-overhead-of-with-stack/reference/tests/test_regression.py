import sys
sys.path.insert(0, ".")
from proftune.budget import select_config_for_budget

def test_tight_budget_disables_expensive_flags():
    cfg = select_config_for_budget({}, 3000)
    assert cfg["with_stack"] is False, "tight budget should disable with_stack"
    assert cfg["record_shapes"] is False, "tight budget should disable record_shapes"

def test_generous_budget_allows_stack():
    cfg = select_config_for_budget({}, 50000)
    assert cfg["with_stack"] is True, "generous budget should allow with_stack"
