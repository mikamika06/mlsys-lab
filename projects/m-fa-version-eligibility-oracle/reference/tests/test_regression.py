from fa_oracle.planner import plan_upgrade

def test_plan_upgrade_basic():
    stack = {"compute_capability": "8.0", "flash_attn_version": "2.5.8"}
    res = plan_upgrade(stack, "fa2")
    assert res["action"] == "none"

def test_plan_upgrade_hardware():
    stack = {"compute_capability": "7.5", "flash_attn_version": "2.5.8"}
    res = plan_upgrade(stack, "fa3")
    assert res["action"] == "hardware_upgrade"
    assert res["target_gpu"] == "H100"
