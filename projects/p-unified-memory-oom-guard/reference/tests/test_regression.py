import sys
sys.path.insert(0, ".")
from guard.memory import measure_footprint
from guard.predictor import predict_peak
from guard.limiter import RuntimeLimiter, degrade_gracefully

def test_prediction_accuracy():
    cfg = {"context_length": 1024, "num_layers": 16, "hidden_size": 2048, "bytes_per_param": 2}
    measured = measure_footprint(cfg)
    predicted = predict_peak(cfg)
    diff = abs(predicted - measured) / measured
    assert diff <= 0.15, f"Prediction error {diff} exceeds tolerance"

def test_limiter_triggers():
    limiter = RuntimeLimiter(max_memory=1000)
    assert limiter.check_and_apply(1500) == "degrade"
    assert limiter.check_and_apply(500) == "allow"

def test_degrade_reduces_footprint():
    cfg = {"context_length": 2048, "num_layers": 16}
    deg = degrade_gracefully(cfg, 1000)
    assert deg["context_length"] <= 1024
