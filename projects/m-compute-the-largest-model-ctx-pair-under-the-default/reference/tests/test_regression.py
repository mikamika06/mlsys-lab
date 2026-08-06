from runner_limits.diagnose import diagnose_log


def test_diagnose_metal_failure():
    log = "Error: Metal buffer allocation failed during command buffer commit."
    assert diagnose_log(log) == "metal_alloc_failure"


def test_diagnose_oom():
    log = "Killed: 9 process exceeded memory limits."
    assert diagnose_log(log) == "oom_kill"
