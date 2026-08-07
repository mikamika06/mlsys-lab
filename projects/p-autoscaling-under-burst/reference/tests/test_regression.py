import sys
sys.path.insert(0, ".")
from scaler.cost import measure_cold_start_cost
from scaler.queue import simulate_queue
from scaler.policy import should_admit
from scaler.predictor import predict_scaling_action

def test_cost_calculation():
    res = measure_cold_start_cost(1000, 100, 10)
    assert res["total_time"] == 20.0

def test_queue_simulation():
    res = simulate_queue(10, 15, 1, 3)
    assert res["max_queue"] > 0

def test_policy_admission():
    assert should_admit(5, 10, 0.5) is True
    assert should_admit(15, 10, 0.5) is False

def test_predictor_action():
    res = predict_scaling_action(50, 2, 30, 10)
    assert res["action"] == "scale_up"
