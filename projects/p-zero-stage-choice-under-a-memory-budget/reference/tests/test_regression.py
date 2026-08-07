import sys
import pytest

sys.path.insert(0, ".")
from zero_planner.estimator import ZeroEstimator
from zero_planner.planner import ZeroPlanner


def test_estimator_zero_stages_progression():
    est = ZeroEstimator(num_params=1000000)
    w = 8
    act = 1000000
    m1 = est.memory_zero1(w, act)
    m2 = est.memory_zero2(w, act)
    m3 = est.memory_zero3(w, act)
    assert m1 > m2 > m3


def test_estimator_comm_bytes():
    est = ZeroEstimator(num_params=1000000)
    c1 = est.comm_bytes_per_step(1, 4)
    c2 = est.comm_bytes_per_step(2, 4)
    c3 = est.comm_bytes_per_step(3, 4)
    assert c1 == c2
    assert c3 > c1


def test_planner_doubled_gpus_reduces_memory():
    planner = ZeroPlanner(num_params=10000000)
    res = planner.predict_doubled_gpus(current_world_size=4, stage=2, act_mem_per_gpu=5000000)
    assert res["new_world_size"] == 8
    assert res["memory_saved_bytes"] > 0
