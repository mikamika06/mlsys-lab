from mixplan.budget import choose_quant_under_budget

def test_budget_selection():
    tensors = [{"name": "blk.0.weight", "numel": 100, "shape": [10, 10]}]
    res = choose_quant_under_budget(tensors, 1000, ["Q4_0", "F32"])
    assert res in ["Q4_0", "F32"]
