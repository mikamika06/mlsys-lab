from ensemble.wiring import build_and_validate_wiring

def test_wiring_structure():
    cfg = {
        "step1": {"name": "a", "inputs": ["x"], "outputs": ["y"]},
        "step2": {"name": "b", "inputs": ["y"], "outputs": ["z"]}
    }
    res = build_and_validate_wiring(cfg)
    assert "inputs" in res
    assert "outputs" in res
    assert len(res["steps"]) == 2
