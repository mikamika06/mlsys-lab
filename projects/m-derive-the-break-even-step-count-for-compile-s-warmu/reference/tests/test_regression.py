from compile_peft.adapters import count_recompilations

def test_recompilations_basic():
    scenario = {"adapters": ["adapter_a", "adapter_b"], "shared_base": True}
    assert count_recompilations(scenario) == 2

def test_recompilations_identical():
    scenario = {"adapters": ["adapter_a", "adapter_a"], "shared_base": True}
    assert count_recompilations(scenario) == 1
