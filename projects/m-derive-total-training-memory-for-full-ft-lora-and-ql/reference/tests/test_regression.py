import sys
sys.path.insert(0, ".")
from finetune.budget import classify_hardware_budgets

def test_budget_classification():
    budgets = {"small": 10 * 1024 * 1024 * 1024, "large": 1000 * 1024 * 1024 * 1024}
    res = classify_hardware_budgets(1_000_000_000, budgets)
    assert isinstance(res, list)
    assert len(res) > 0
    for m in res:
        assert m in ["full", "lora", "qlora"]
