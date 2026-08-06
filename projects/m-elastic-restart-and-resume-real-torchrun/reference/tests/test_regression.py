import sys
sys.path.insert(0, ".")
from elastic.parser import parse_nccl_log
from elastic.rendezvous import compute_membership
from elastic.resume import verify_resume_state

def test_parse_nccl_log_valid():
    log = "[rank 3] Watchdog caught collective execution timeout during operation allreduce"
    res = parse_nccl_log(log)
    assert res["failed_rank"] == 3
    assert res["timeout_op"] == "allreduce"
    assert res["has_timeout"] is True

def test_compute_membership_basic():
    res = compute_membership(4, [1], 3)
    assert res["world_size"] == 3
    assert res["active_ranks"] == [0, 2, 3]
    assert res["mapping"] == {0: 0, 2: 1, 3: 2}

def test_verify_resume_state_correct():
    state = {"step": 100, "model_weights": {"layer.weight": [0.1, 0.2]}}
    assert verify_resume_state(state, 100) is True

def test_verify_resume_state_incorrect_step():
    state = {"step": 99, "model_weights": {"layer.weight": [0.1, 0.2]}}
    assert verify_resume_state(state, 100) is False
