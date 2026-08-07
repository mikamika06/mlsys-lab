import sys
sys.path.insert(0, ".")
from triton_verify.parser import compute_concurrency_ceiling
from triton_verify.scaling import compute_scaling_efficiency
from triton_verify.classifier import classify_error

def test_concurrency_ceiling_basic():
    cfg = {"max_batch_size": 4, "instance_group": [{"count": 2}], "dynamic_range_limit": 8}
    assert compute_concurrency_ceiling(cfg) == 64

def test_scaling_efficiency_bounds():
    configs = [{"instance_group": [{"count": 1}]}, {"instance_group": [{"count": 2}]}]
    throughputs = [100.0, 180.0]
    eff = compute_scaling_efficiency(configs, throughputs)
    assert len(eff) == 2
    assert eff[0] == 1.0

def test_classifier_accuracy():
    err = "Error: dynamic shape violation detected in tensor dimension"
    assert classify_error(err) == "DYNAMIC_SHAPE_ERROR"
